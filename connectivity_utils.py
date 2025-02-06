import numpy as np
from sklearn import neighbors
import torch

def _compute_connectivity(positions, radius, add_self_edges):
    """
    Compute connectivity for a set of node positions.

    Args:
      positions: A numpy array of shape [num_nodes, num_dims] representing node positions.
      radius: A float specifying the connectivity radius.
      add_self_edges: Boolean indicating whether self-edges (i.e. edges from a node to itself)
                      should be included.

    Returns:
      senders: A numpy array of sender indices with shape [num_edges].
      receivers: A numpy array of receiver indices with shape [num_edges].
    """
    tree = neighbors.KDTree(positions)
    # For each position, find all indices of nodes within the radius.
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    # For each node i, repeat i as many times as there are neighbors.
    senders = np.repeat(np.arange(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)
    
    if not add_self_edges:
        # Remove edges where the sender and receiver are identical.
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]
    
    return senders, receivers


def _compute_connectivity_for_batch(positions, n_node, radius, add_self_edges):
    """
    Compute connectivity for a batch of graphs.

    Args:
      positions: A numpy array of shape [total_num_nodes, num_dims] containing positions for all nodes
                 in the batch.
      n_node: A 1D numpy array (or list) of length [batch_size] containing the number of nodes in each graph.
      radius: A float specifying the connectivity radius.
      add_self_edges: Boolean indicating whether self-edges should be included.

    Returns:
      senders: A concatenated numpy array of sender indices for the batch.
      receivers: A concatenated numpy array of receiver indices for the batch.
      n_edge: A numpy array of length [batch_size] with the number of edges for each graph.
    """
    # Split positions into a list (one per graph) based on n_node.
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    
    senders_list = []
    receivers_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0

    for positions_graph in positions_per_graph_list:
        senders_graph, receivers_graph = _compute_connectivity(positions_graph, radius, add_self_edges)
        num_edges_graph = len(senders_graph)
        n_edge_list.append(num_edges_graph)
        
        # Adjust indices because the graphs will later be concatenated.
        senders_list.append(senders_graph + num_nodes_in_previous_graphs)
        receivers_list.append(receivers_graph + num_nodes_in_previous_graphs)
        
        num_nodes_in_previous_graphs += positions_graph.shape[0]

    senders = np.concatenate(senders_list, axis=0).astype(np.int64)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int64)
    n_edge = np.array(n_edge_list, dtype=np.int64)

    return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(positions, n_node, radius, add_self_edges=True):
    """
    Compute connectivity for a batch of graphs, accepting PyTorch tensors as inputs and returning
    PyTorch tensors. This function is used to determine graph connectivity based on a radius.

    Args:
      positions: A torch.Tensor of shape [total_num_nodes, num_dims] (expected to be on CPU).
      n_node: A torch.Tensor or list containing the number of nodes for each graph.
      radius: A float specifying the connectivity radius.
      add_self_edges: Boolean indicating whether to include self-edges.

    Returns:
      senders: A torch.Tensor of sender indices.
      receivers: A torch.Tensor of receiver indices.
      n_edge: A torch.Tensor of the number of edges per graph.
    """
    # Convert inputs to numpy arrays if they are torch.Tensors.
    if isinstance(positions, torch.Tensor):
        positions_np = positions.detach().cpu().numpy()
    else:
        positions_np = np.array(positions)
    if isinstance(n_node, torch.Tensor):
        n_node_np = n_node.detach().cpu().numpy()
    else:
        n_node_np = np.array(n_node)
    
    senders_np, receivers_np, n_edge_np = _compute_connectivity_for_batch(positions_np, n_node_np, radius, add_self_edges)
    
    # Convert outputs back to torch tensors.
    senders = torch.tensor(senders_np, dtype=torch.long)
    receivers = torch.tensor(receivers_np, dtype=torch.long)
    n_edge = torch.tensor(n_edge_np, dtype=torch.long)
    return senders, receivers, n_edge
