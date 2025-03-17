import numpy as np
from sklearn.neighbors import KDTree


def extend_positions(positions, box_size):
    """
    Extend positions with periodic copies (ghost cells) for a hypercubic box.
    
    Args:
        positions (np.ndarray): Array of shape [N, d] with original positions.
        box_size (float): The length of the simulation box in each dimension.
        
    Returns:
        extended_positions (np.ndarray): Array of shape [N * 3^d, d] with ghost copies.
        mapping (np.ndarray): Array of shape [N * 3^d] mapping each ghost copy to its original index.
    """
    N, d = positions.shape
    # Create shifts for each dimension: [-box_size, 0, box_size]
    shift_values = [-box_size, 0, box_size]
    # Create a grid of shifts. The result has shape [3, 3, ..., 3, d] which we then reshape.
    grids = np.meshgrid(*([shift_values] * d))
    shifts = np.stack(grids, axis=-1).reshape(-1, d)  # shape: [3^d, d]
    
    extended_positions = []
    mapping = []
    for shift in shifts:
        extended_positions.append(positions + shift)
        mapping.append(np.arange(N))
    extended_positions = np.concatenate(extended_positions, axis=0)
    mapping = np.concatenate(mapping, axis=0)
    return extended_positions, mapping
    

def compute_connectivity(positions, num_neighbors, add_self_edges=True, box_size=None):
    """
    Computes the connectivity for a single graph given node positions using k-nearest neighbors, with support for periodic boundaries via extended domain.
    
    Args:
        positions (np.ndarray): Array of shape [num_nodes, num_dims] representing node positions.
        num_neighbors (int): The number of nearest neighbors to consider for each node.
        add_self_edges (bool): Whether to include self edges (node connected to itself).
        box_size (float or None): If provided, the positions are extended periodically with this box size.
        
    Returns:
        senders (np.ndarray): 1D array of sender node indices.
        receivers (np.ndarray): 1D array of receiver node indices.
    """
    if box_size is not None:
        extended_positions, mapping = extend_positions(positions, box_size)
        tree = KDTree(extended_positions)
        # Query using original positions against the extended set.
        distances, indices = tree.query(positions, k=num_neighbors)
        # Map the indices from the extended set back to original indices.
        neighbor_indices = mapping[indices]
    else:
        tree = KDTree(positions)
        distances, indices = tree.query(positions, k = num_neighbors)
    
    num_nodes = positions.shape[0]
    # Each node now connects to num_neighbor nodes.
    senders = np.repeat(np.arange(num_nodes), num_neighbors)
    receivers = indices.flatten()

    if not add_self_edges:
        # Remove self edges (where sender equals receiver).
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]
    
    return senders, receivers

def compute_connectivity_for_batch(positions, n_node, num_neighbors, add_self_edges=True, box_size=None):
    """
    Computes connectivity for a batch of graphs using k-nearest neighbors with optional periodicity.
    
    Args:
        positions (np.ndarray): Array of shape [total_nodes, num_dims] for all graphs in the batch.
        n_node (np.ndarray or list): 1D array of integers representing the number of nodes per graph.
        num_neighbors (int): The number of nearest neighbors to consider for each node.
        add_self_edges (bool): Whether to include self edges.
        box_size (float or None): If provided, uses periodic extension with this box size.
        
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
        s, r = compute_connectivity(pos, num_neighbors, add_self_edges, box_size=box_size)
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