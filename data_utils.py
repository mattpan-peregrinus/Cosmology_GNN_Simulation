import torch
import torch_geometric as pyg
from torch_geometric.nn import knn_graph
from torch_cluster import knn
import torch_scatter
import math

def extend_positions_torch(positions, box_size):
    """
    Extend positions with periodic copies for a hypercubic simulation box.
    
    Args:
        positions (torch.Tensor): Tensor of shape [N, d] with original positions.
        box_size (float): The length of the simulation box in each dimension.
        
    Returns:
        extended_positions (torch.Tensor): Tensor of shape [N * 3^d, d] with ghost copies.
        mapping (torch.Tensor): Tensor of shape [N * 3^d] mapping each ghost copy to its original index.
    """
    N, d = positions.size()
    if isinstance(box_size, list):
        box_size = float(box_size[0])
    # Create shift values for each dimension: [-box_size, 0, box_size]
    shift_values = [-box_size, 0, box_size]
    # Create all combinations using torch.cartesian_prod.
    shifts = torch.cartesian_prod(*[torch.tensor(shift_values, device=positions.device, dtype=torch.float32) for _ in range(d)])
    # shifts shape: [3^d, d]
    extended_positions = []
    mapping = []
    for shift in shifts:
        # Add shift to positions; shift is [d], so unsqueeze to [1, d] for broadcasting.
        extended_positions.append(positions + shift.unsqueeze(0))
        mapping.append(torch.arange(N, device=positions.device))
    extended_positions = torch.cat(extended_positions, dim=0) 
    mapping = torch.cat(mapping, dim=0) 
    return extended_positions, mapping


def generate_noise(position_seq, noise_std):
    position_seq = position_seq.float()
    """Generate random-walk noise for a trajectory."""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq, dtype = torch.float32) * (noise_std / (time_steps ** 0.5))
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise, dtype = torch.float32)[:, 0:1], position_noise), dim=1)
    return position_noise

def preprocess(particle_type, position_seq, target_position, metadata, noise_std, num_neighbors):
    """Preprocess a trajectory and construct a PyG Data object."""
    position_seq = position_seq.float()
    if target_position is not None:
        target_position = target_position.float()
    
    # Apply noise
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    # Compute velocities
    recent_position = position_seq[:, -1]
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]

    # Construct the graph via k-nearest neighbors with periodicity if desired.
    if "box_size" in metadata and metadata["box_size"] is not None:
        box_size = metadata["box_size"]
        if isinstance(box_size, list):
            box_size = float(box_size[0])
        
        extended_positions, mapping = extend_positions_torch(recent_position, box_size)
        edge_index = knn(extended_positions, recent_position, num_neighbors)
        edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index[0] = mapping[edge_index[0]]
        edge_index[1] = mapping[edge_index[1]]
    else:
        # Non-periodic connectivity.
        edge_index = knn_graph(
            recent_position,
            k=num_neighbors,
            loop=True
        )

    # Node-level features
    vel_mean = torch.tensor(metadata["vel_mean"], dtype=torch.float32)
    vel_std = torch.tensor(metadata["vel_std"], dtype=torch.float32)
    normal_velocity_seq = (velocity_seq - vel_mean) / torch.sqrt(vel_std**2 + noise_std**2)
    boundary = torch.tensor(metadata["bounds"], dtype=torch.float32)
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    distance_to_boundary = torch.clip(distance_to_boundary, -1.0, 1.0)
    
    if particle_type is None:
        # Use flattened velocities and boundary distances as node features
        node_features = torch.cat((
            normal_velocity_seq.reshape(normal_velocity_seq.size(0), -1),
            distance_to_boundary
        ), dim=-1)
    else:
        node_features = particle_type.float32
        
        
    # Edge-level features
    dim = recent_position.size(-1)
    senders = edge_index[0]
    receivers = edge_index[1]
    
    # Add checks to ensure indices are within bounds
    assert torch.max(senders) < len(recent_position), f"Max sender index {torch.max(senders)} >= {len(recent_position)}"
    assert torch.max(receivers) < len(recent_position), f"Max receiver index {torch.max(receivers)} >= {len(recent_position)}"

    edge_displacement = recent_position[senders] - recent_position[receivers]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # Ground truth for training (acceleration)
    if target_position is not None:
        if target_position.dim() > 1:
            target_position = target_position[0]    # Take the first time-step 
        
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        
        acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
        acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
        acceleration = (acceleration - acc_mean) / torch.sqrt(acc_std**2 + noise_std**2)
    else:
        acceleration = None

    # Build a PyG Data object
    from torch_geometric.data import Data
    graph = Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y=acceleration.float() if acceleration is not None else None,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1)
    )
    return graph
