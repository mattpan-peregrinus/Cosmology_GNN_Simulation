import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing


###########################################
# Helper functions
###########################################

def relu(x):
    return F.relu(x)

def build_mlp(hidden_size: int, num_hidden_layers: int, output_size: int) -> nn.Module:
    """
    Builds an MLP. We use LazyLinear for the first layer so that the input dimension
    is inferred upon the first call.

    The MLP architecture is:
      LazyLinear/Linear -> ReLU (repeated num_hidden_layers times)
      followed by a final Linear layer to output_size.
    """
    layers = []
    for i in range(num_hidden_layers):
        if i == 0:
            layers.append(nn.LazyLinear(hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU(inplace=True))
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


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


class InteractionNetwork(MessagePassing):
    """
    One round of message passing: updates edge features using the sender and
    receiver node features, aggregates messages (by summing) at the target nodes,
    and updates node features.

    (Residual additions are applied in the outer module.)
    """

    def __init__(self, node_model: nn.Module,
                 edge_model: nn.Module,
                 aggr: str = 'add'):
        super().__init__(aggr=aggr)
        self.node_model = node_model
        self.edge_model = edge_model

    def forward(self, data: Data) -> Data:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        src, dest = edge_index[0], edge_index[1]
        if edge_attr is None:
            raise ValueError("edge_attr must not be None in InteractionNetwork")
        # For each edge, compute updated edge features.
        edge_input = torch.cat([x[src], x[dest], edge_attr], dim=-1)
        updated_edge = self.edge_model(edge_input)
        # For each node, aggregate incoming edge messages.
        aggregated_messages = self.propagate(edge_index, x=x, edge_attr=updated_edge)
        # Concatenate node features with aggregated message.
        node_input = torch.cat([x, aggregated_messages], dim=-1)
        # Update node features.
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
                 output_size: int):
        super(EncodeProcessDecode, self).__init__()
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size

        # Helper: Build an MLP followed by a LayerNorm.
        def build_mlp_with_layer_norm():
            mlp = build_mlp(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
            return nn.Sequential(mlp, nn.LayerNorm(latent_size))

        self.encoder = GraphIndependent(
            node_model=build_mlp_with_layer_norm(),
            edge_model=build_mlp_with_layer_norm()
        )

        self.processor = nn.ModuleList([
            InteractionNetwork(
                edge_model=build_mlp_with_layer_norm(),
                node_model=build_mlp_with_layer_norm()
            )
            for _ in range(num_message_passing_steps)
        ])

        self.decoder = build_mlp(mlp_hidden_size, mlp_num_hidden_layers, output_size)

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
        latent_graph_0 = self.encoder(input_graph)
        return latent_graph_0

    def _process(self, latent_graph: Data) -> Data:
        for processor in self.processor:
            new_latent_graph = processor(latent_graph)
            # Add residual connections.
            latent_graph.x = latent_graph.x + new_latent_graph.x
            latent_graph.edge_attr = latent_graph.edge_attr + new_latent_graph.edge_attr
        return latent_graph

    def _decode(self, latent_graph: Data) -> torch.Tensor:
        # Decode node latent features.
        return self.decoder(latent_graph.x)
