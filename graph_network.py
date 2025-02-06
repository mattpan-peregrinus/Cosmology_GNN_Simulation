import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_add  # pip install torch-scatter

###########################################
# Helper functions for building MLPs
###########################################

def build_mlp(hidden_size: int, num_hidden_layers: int, output_size: int) -> nn.Module:
    """
    Builds an MLP. Uses nn.LazyLinear for the first layer so that the input
    dimension is inferred on the first call.
    
    If num_hidden_layers is 0, then the network is a single linear mapping.
    Otherwise, the network is: LazyLinear -> ReLU -> (Linear -> ReLU)^(num_hidden_layers-1)
    -> Linear.
    """
    layers = []
    if num_hidden_layers == 0:
        layers.append(nn.LazyLinear(output_size))
    else:
        layers.append(nn.LazyLinear(hidden_size))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


def build_mlp_with_layer_norm(hidden_size: int, num_hidden_layers: int, output_size: int) -> nn.Module:
    """
    Builds an MLP (see build_mlp) and then applies a LayerNorm over the last dimension.
    """
    mlp = build_mlp(hidden_size, num_hidden_layers, output_size)
    return nn.Sequential(mlp, nn.LayerNorm(output_size))


###########################################
# Graph Modules
###########################################

class GraphIndependent(nn.Module):
    """
    Applies independent encoders to node and edge features.
    
    If the input Data object has a 'globals' attribute, it is assumed that
    the globals have already been broadcast and concatenated to the nodes.
    """
    def __init__(self, node_model: nn.Module, edge_model: nn.Module):
        super(GraphIndependent, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model

    def forward(self, data: Data) -> Data:
        # Process node features.
        x = self.node_model(data.x)
        # Process edge features (if available).
        if data.edge_attr is not None:
            edge_attr = self.edge_model(data.edge_attr)
        else:
            edge_attr = None
        # Create a new Data object.
        new_data = Data(x=x, edge_index=data.edge_index, edge_attr=edge_attr)
        if hasattr(data, 'globals'):
            new_data.globals = data.globals
        return new_data


class InteractionNetwork(nn.Module):
    """
    One round of message passing: updates edge features using the sender and
    receiver node features, aggregates messages (by summing) at the target nodes,
    and updates node features.
    
    (Residual additions are applied in the outer module.)
    """
    def __init__(self, node_model: nn.Module, edge_model: nn.Module, reducer=scatter_add):
        super(InteractionNetwork, self).__init__()
        self.node_model = node_model
        self.edge_model = edge_model
        self.reducer = reducer

    def forward(self, data: Data) -> Data:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dest = edge_index[0], edge_index[1]
        if edge_attr is None:
            raise ValueError("edge_attr must not be None in InteractionNetwork")
        # For each edge, compute updated edge features.
        edge_input = torch.cat([x[src], x[dest], edge_attr], dim=-1)
        updated_edge = self.edge_model(edge_input)
        # For each node, aggregate incoming edge messages.
        num_nodes = x.size(0)
        aggregated_messages = self.reducer(updated_edge, dest, dim=0, dim_size=num_nodes)
        # Update node features.
        node_input = torch.cat([x, aggregated_messages], dim=-1)
        updated_node = self.node_model(node_input)
        # Return an updated graph.
        new_data = Data(x=updated_node, edge_index=edge_index, edge_attr=updated_edge)
        if hasattr(data, 'globals'):
            new_data.globals = data.globals
        return new_data


###########################################
# Main Encode-Process-Decode Module
###########################################

class EncodeProcessDecode(nn.Module):
    """
    Encode–Process–Decode module for learnable simulation.
    
    This module assumes that an encoder has already built a graph with connectivity
    and features. It then:
      1. Encodes the graph into a latent graph.
      2. Processes the latent graph with several rounds of message passing (with residuals).
      3. Decodes the latent node representations into an output.
    """
    def __init__(self,
                 latent_size: int,
                 mlp_hidden_size: int,
                 mlp_num_hidden_layers: int,
                 num_message_passing_steps: int,
                 output_size: int,
                 reducer = scatter_add):
        super(EncodeProcessDecode, self).__init__()
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer

        self.networks_builder()

    def networks_builder(self):
        # Helper to build an MLP with layer norm.
        def build_mlp_with_ln():
            return build_mlp_with_layer_norm(
                self._mlp_hidden_size,
                self._mlp_num_hidden_layers,
                self._latent_size
            )
        # Encoder: independently encode nodes and edges.
        self._encoder_network = GraphIndependent(
            node_model=build_mlp_with_ln(),
            edge_model=build_mlp_with_ln()
        )
        # Processor networks: a list of InteractionNetworks (unshared parameters).
        self._processor_networks = nn.ModuleList([
            InteractionNetwork(
                node_model=build_mlp_with_ln(),
                edge_model=build_mlp_with_ln(),
                reducer=self._reducer
            ) for _ in range(self._num_message_passing_steps)
        ])
        # Decoder: decodes node latent features to the desired output.
        self._decoder_network = build_mlp(
            self._mlp_hidden_size,
            self._mlp_num_hidden_layers,
            self._output_size
        )

    def forward(self, input_graph: Data) -> torch.Tensor:
        latent_graph_0 = self._encode(input_graph)
        latent_graph_m = self._process(latent_graph_0)
        return self._decode(latent_graph_m)

    def _encode(self, input_graph: Data) -> Data:
        # If globals exist, broadcast them to every node.
        if hasattr(input_graph, 'globals') and input_graph.globals is not None:
            global_feat = input_graph.globals  # Expected shape: [global_dim]
            global_broadcast = global_feat.unsqueeze(0).expand(input_graph.x.size(0), -1)
            x = torch.cat([input_graph.x, global_broadcast], dim=-1)
            input_graph = Data(x=x, edge_index=input_graph.edge_index, edge_attr=input_graph.edge_attr)
            input_graph.globals = None  # Remove globals once broadcast.
        latent_graph_0 = self._encoder_network(input_graph)
        return latent_graph_0

    def _process(self, latent_graph_0: Data) -> Data:
        latent_graph_prev = latent_graph_0
        latent_graph = latent_graph_0
        for processor in self._processor_networks:
            latent_graph = processor(latent_graph_prev)
            # Add residual connections.
            latent_graph.x = latent_graph.x + latent_graph_prev.x
            latent_graph.edge_attr = latent_graph.edge_attr + latent_graph_prev.edge_attr
            latent_graph_prev = latent_graph
        return latent_graph

    def _decode(self, latent_graph: Data) -> torch.Tensor:
        # Decode node latent features.
        return self._decoder_network(latent_graph.x)
