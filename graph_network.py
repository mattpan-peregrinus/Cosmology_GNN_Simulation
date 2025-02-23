import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_add  
from torch_geometric.nn import SAGEConv

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



class GraphSAGENetwork(nn.Module):
    """
    GraphSAGE implementation for message passing.
    Uses multiple layers of GraphSAGE convolutions with residual connections.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(GraphSAGENetwork, self).__init__()
        self.sage_conv1 = SAGEConv(input_size, hidden_size)
        self.sage_conv2 = SAGEConv(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index
        
        # First GraphSAGE layer
        h = F.relu(self.sage_conv1(x, edge_index))
        
        # Second GraphSAGE layer with residual connection
        h = self.sage_conv2(h, edge_index)
        h = self.layer_norm(h + x if x.size(-1) == h.size(-1) else h)
        
        # Return updated graph
        new_data = Data(x=h, edge_index=edge_index)
        if hasattr(data, 'globals'):
            new_data.globals = data.globals
        return new_data
    

###########################################
# Main Encode-Process-Decode Module
###########################################

class EncodeProcessDecode(nn.Module):
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

        self.networks_builder()

    def networks_builder(self):
        # Encoder: independently encode nodes
        self._encoder_network = GraphIndependent(
            node_model=build_mlp_with_layer_norm(
                self._mlp_hidden_size,
                self._mlp_num_hidden_layers,
                self._latent_size
            ),
            edge_model=None  # GraphSAGE doesn't use edge features
        )
        
        # Processor networks: a list of GraphSAGE networks
        self._processor_networks = nn.ModuleList([
            GraphSAGENetwork(
                input_size=self._latent_size,
                hidden_size=self._mlp_hidden_size,
                output_size=self._latent_size
            ) for _ in range(self._num_message_passing_steps)
        ])
        
        # Decoder remains the same
        self._decoder_network = build_mlp(
            self._mlp_hidden_size,
            self._mlp_num_hidden_layers,
            self._output_size
        )

    def _process(self, latent_graph_0: Data) -> Data:
        latent_graph = latent_graph_0
        for processor in self._processor_networks:
            latent_graph = processor(latent_graph)
        return latent_graph
