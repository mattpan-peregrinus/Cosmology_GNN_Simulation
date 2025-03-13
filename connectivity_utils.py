import numpy as np
from sklearn.neighbors import KDTree

def compute_connectivity(positions, radius, add_self_edges=True):
    """
    Computes the connectivity for a single graph given node positions.
    
    Args:
        positions (np.ndarray): Array of shape [num_nodes, num_dims] representing node positions.
        radius (float): Maximum distance to consider two nodes connected.
        add_self_edges (bool): Whether to include self edges (node connected to itself).
        
    Returns:
        senders (np.ndarray): 1D array of sender node indices.
        receivers (np.ndarray): 1D array of receiver node indices.
    """
    # Build a KDTree for fast neighbor lookup.
    tree = KDTree(positions)
    # For each node, find indices of all nodes within 'radius'.
    neighbors_list = tree.query_radius(positions, r=radius)
    
    num_nodes = positions.shape[0]
    # Repeat each node index for the number of neighbors it has.
    senders = np.repeat(np.arange(num_nodes), [len(neighbors) for neighbors in neighbors_list])
    receivers = np.concatenate(neighbors_list, axis=0)

    if not add_self_edges:
        # Remove self edges (where sender equals receiver).
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]
    
    return senders, receivers

def compute_connectivity_for_batch(positions, n_node, radius, add_self_edges=True):
    """
    Computes connectivity for a batch of graphs.
    
    Args:
        positions (np.ndarray): Array of shape [total_nodes, num_dims] for all graphs in the batch.
        n_node (np.ndarray or list): 1D array of integers representing the number of nodes per graph.
        radius (float): Maximum distance to consider two nodes connected.
        add_self_edges (bool): Whether to include self edges.
        
    Returns:
        senders (np.ndarray): 1D array of sender node indices for the batch.
        receivers (np.ndarray): 1D array of receiver node indices for the batch.
        n_edge (np.ndarray): 1D array indicating the number of edges per graph.
    """
    # Split positions into separate graphs based on n_node.
    positions_split = np.split(positions, np.cumsum(n_node)[:-1], axis=0)
    
    senders_list = []
    receivers_list = []
    n_edge_list = []
    offset = 0
    
    # Process each graph individually.
    for pos in positions_split:
        s, r = compute_connectivity(pos, radius, add_self_edges)
        num_edges = len(s)
        n_edge_list.append(num_edges)
        
        # Adjust indices by offset.
        senders_list.append(s + offset)
        receivers_list.append(r + offset)
        offset += pos.shape[0]
    
    # Concatenate results for the entire batch.
    senders = np.concatenate(senders_list, axis=0).astype(np.int64)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int64)
    n_edge = np.array(n_edge_list, dtype=np.int64)
    
    return senders, receivers, n_edge