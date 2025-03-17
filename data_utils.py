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
    # Create shift values for each dimension: [-box_size, 0, box_size]
    shift_values = [-box_size, 0, box_size]
    # Create all combinations using torch.cartesian_prod.
    shifts = torch.cartesian_prod(*[torch.tensor(shift_values, device=positions.device, dtype=positions.dtype) for _ in range(d)])
    # shifts shape: [3^d, d]
    extended_positions = []
    mapping = []
    for shift in shifts:
        # Add shift to positions; shift is [d], so unsqueeze to [1, d] for broadcasting.
        extended_positions.append(positions + shift.unsqueeze(0))
        mapping.append(torch.arange(N, device=positions.device))
    extended_positions = torch.cat(extended_positions, dim=0)  # Shape: [N * 3^d, d]
    mapping = torch.cat(mapping, dim=0)  # Shape: [N * 3^d]
    return extended_positions, mapping


def generate_noise(position_seq, noise_std):
    """Generate random-walk noise for a trajectory."""
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]
    time_steps = velocity_seq.size(1)
    velocity_noise = torch.randn_like(velocity_seq) * (noise_std / (time_steps ** 0.5))
    velocity_noise = velocity_noise.cumsum(dim=1)
    position_noise = velocity_noise.cumsum(dim=1)
    position_noise = torch.cat((torch.zeros_like(position_noise)[:, 0:1], position_noise), dim=1)
    return position_noise

def preprocess(particle_type, position_seq, target_position, metadata, noise_std, num_neighbors):
    """Preprocess a trajectory and construct a PyG Data object."""
    # Apply noise
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    # Compute velocities
    recent_position = position_seq[:, -1]
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]

    # Construct the graph via k-nearest neighbors with periodicity if desired.
    if "box_size" in metadata and metadata["box_size"] is not None:
        extended_positions, mapping = extend_positions_torch(recent_position, metadata["box_size"])
        edge_index = knn(recent_position, extended_positions, num_neighbors)
        edge_index[1] = mapping[edge_index[1]]
    else:
        # Non-periodic connectivity.
        edge_index = knn_graph(
            recent_position,
            k=num_neighbors,
            loop=True
        )

    # Node-level features
    normal_velocity_seq = (velocity_seq - torch.tensor(metadata["vel_mean"])) \
        / torch.sqrt(torch.tensor(metadata["vel_std"]) ** 2 + noise_std ** 2)
    boundary = torch.tensor(metadata["bounds"])
    distance_to_lower_boundary = recent_position - boundary[:, 0]
    distance_to_upper_boundary = boundary[:, 1] - recent_position
    distance_to_boundary = torch.cat((distance_to_lower_boundary, distance_to_upper_boundary), dim=-1)
    distance_to_boundary = torch.clip(distance_to_boundary, -1.0, 1.0)

    # Edge-level features
    dim = recent_position.size(-1)
    senders = edge_index[0].unsqueeze(-1).expand(-1, dim)
    receivers = edge_index[1].unsqueeze(-1).expand(-1, dim)
    edge_displacement = recent_position[senders] - recent_position[receivers]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)

    # Ground truth for training (acceleration)
    if target_position is not None:
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position + position_noise[:, -1] - recent_position
        acceleration = next_velocity - last_velocity
        acceleration = (acceleration - torch.tensor(metadata["acc_mean"])) \
            / torch.sqrt(torch.tensor(metadata["acc_std"]) ** 2 + noise_std ** 2)
    else:
        acceleration = None

    # Build a PyG Data object
    from torch_geometric.data import Data
    graph = Data(
        x=particle_type,  # particle type
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y=acceleration,
        pos=torch.cat((velocity_seq.reshape(velocity_seq.size(0), -1), distance_to_boundary), dim=-1)
    )
    return graph
