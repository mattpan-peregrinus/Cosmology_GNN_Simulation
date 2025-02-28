import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_max
from torch_geometric.nn import SAGEConv

###########################################
# Helper functions for building MLPs
###########################################

def build_mlp(hidden_size: int, num_hidden_layers: int, output_size: int) -> nn.Module:
    """
    Builds an MLP. If num_hidden_layers is 0, then the network is a single linear mapping.
    Otherwise, the network is:
      Linear(hidden_size) -> ReLU -> ... -> Linear(output_size).
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
    def __init__(self, node_model: nn.Module, edge_model: nn.Module = None):
        super().__init__()
        self.node_model = node_model
        self.edge_model = edge_model  

    def forward(self, data: Data) -> Data:
        # Encode node features
        x = self.node_model(data.x)
        # Encode edge features (if we have an edge model)
        if self.edge_model is not None and data.edge_attr is not None:
            edge_attr = self.edge_model(data.edge_attr)
        else:
            edge_attr = data.edge_attr  # Pass through or None
        new_data = Data(x=x, edge_index=data.edge_index, edge_attr=edge_attr)
        if hasattr(data, 'globals'):
            new_data.globals = data.globals
        return new_data

class GraphSAGENetwork(nn.Module):
    """
    GraphSAGE for message passing. Uses multiple layers of SAGEConv with max aggregator
    and a residual-like connection. 
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.sage_conv1 = SAGEConv(in_channels=input_size, out_channels=hidden_size, aggr='max')
        self.sage_conv2 = SAGEConv(in_channels=hidden_size, out_channels=output_size, aggr='max')
        self.layer_norm = nn.LayerNorm(output_size)
        
    def forward(self, data: Data) -> Data:
        x, edge_index = data.x, data.edge_index

        # First SAGE layer
        h = F.relu(self.sage_conv1(x, edge_index))

        # Second SAGE layer, plus a residual if shapes match
        h2 = self.sage_conv2(h, edge_index)
        if x.size(-1) == h2.size(-1):
            h2 = h2 + x  # residual connection
        h2 = self.layer_norm(h2)

        # Return updated graph
        new_data = Data(x=h2, edge_index=edge_index)
        if hasattr(data, 'globals'):
            new_data.globals = data.globals
        return new_data

###########################################
# Main Encode-Process-Decode Module
###########################################

class EncodeProcessDecode(nn.Module):
    """
    Encode-Process-Decode architecture with GraphSAGE in the 'process' stage.
    """
    def __init__(
        self,
        latent_size: int,
        mlp_hidden_size: int,
        mlp_num_hidden_layers: int,
        num_message_passing_steps: int,
        output_size: int,
        reducer = lambda x, idx, dim_size: scatter_max(x, idx, dim=0, dim_size=dim_size)[0],
    ):
        super().__init__()
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self._reducer = reducer

        self._build_networks()

    def _build_networks(self):
        # Encoder: node_model transforms node features -> latent_size
        self._encoder_network = GraphIndependent(
            node_model=build_mlp_with_layer_norm(
                self._mlp_hidden_size,
                self._mlp_num_hidden_layers,
                self._latent_size
            ),
            edge_model=None 
        )

        # Processor: a list of GraphSAGENetwork steps
        self._processor_networks = nn.ModuleList([
            GraphSAGENetwork(
                input_size=self._latent_size,
                hidden_size=self._mlp_hidden_size,
                output_size=self._latent_size
            ) for _ in range(self._num_message_passing_steps)
        ])

        # Decoder: transforms node latent -> final output size
        self._decoder_network = build_mlp(
            self._mlp_hidden_size,
            self._mlp_num_hidden_layers,
            self._output_size
        )

    def forward(self, input_graph: Data) -> torch.Tensor:
        """
        Full forward pass: encode -> process -> decode.
        Returns a Tensor of shape [num_nodes, output_size].
        """
        latent_graph_0 = self._encode(input_graph)
        latent_graph_m = self._process(latent_graph_0)
        return self._decode(latent_graph_m)

    def _encode(self, input_graph: Data) -> Data:
        # If input_graph has 'globals', they might be broadcast before encoding
        return self._encoder_network(input_graph)

    def _process(self, latent_graph_0: Data) -> Data:
        # Pass through multiple GraphSAGENetwork steps
        latent_graph = latent_graph_0
        for processor in self._processor_networks:
            latent_graph = processor(latent_graph)
        return latent_graph

    def _decode(self, latent_graph: Data) -> torch.Tensor:
        # Decode node latent features
        return self._decoder_network(latent_graph.x)
