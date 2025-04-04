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
    shift_values = [-box_size, 0, box_size]
    shifts = torch.cartesian_prod(*[torch.tensor(shift_values, device=positions.device, dtype=torch.float32) for _ in range(d)])
    extended_positions = []
    mapping = []
    for shift in shifts:
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


def preprocess(particle_type, position_seq, target_position, metadata, noise_std, num_neighbors, temperature_seq, target_temperature=None):
    """Preprocess a trajectory and construct a PyG Data object with both position and temperature target."""
    position_seq = position_seq.float()
    if target_position is not None:
        target_position = target_position.float()
        
    print(f"Original position_seq shape: {position_seq.shape}") # should get [window_size, num_particles, 3]
    
    position_seq = position_seq.permute(1, 0, 2)
    print(f"Transposed position_seq shape: {position_seq.shape}")  # should be [num_particles, window_size, 3]
    
    temperature_seq = temperature_seq.float()
    print(f"Original temperature_seq shape: {temperature_seq.shape}")
    
    if temperature_seq.shape[0] == position_seq.shape[1] and temperature_seq.shape[1] == position_seq.shape[0]:
        temperature_seq = temperature_seq.permute(1, 0, 2)
    print(f"Processed temperature_seq shape: {temperature_seq.shape}")
    
    if target_temperature is not None:
        target_temperature = target_temperature.float()
        print(f"Original target_temperature shape: {target_temperature.shape}")
        if target_temperature.dim() == 3:  # [1, num_particles, 1]
            target_temperature = target_temperature.permute(1, 0, 2)  # -> [num_particles, 1, 1]
            target_temperature = target_temperature.squeeze(1)  # -> [num_particles, 1]
    
    # Apply noise
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    # Compute velocities
    recent_position = position_seq[:, -1] 
    velocity_seq = position_seq[:, 1:] - position_seq[:, :-1]

    # Get recent temperature
    recent_temperature = temperature_seq[:, -1]  # [num_particles, 1]
    temperature_history = temperature_seq[:, :-1]  
        
    print(f"recent_position shape: {recent_position.shape}") # should get [num_particles, 3]
    print(f"velocity_seq shape: {velocity_seq.shape}") # should get [num_particles, window_size-1, 3]
    print(f"temperature_history shape: {temperature_history.shape}") # should get [num_particles, window_size-1, 1]
 
    # Construct the graph via k-nearest neighbors with periodicity if desired
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
        # Non-periodic connectivity
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
    
    print(f"boundary shape: {boundary.shape}") # should get [3, 2]

    # Normalize positions to be between [0, 1] within the simulation box
    if len(boundary.shape) == 2 and boundary.shape[0] == 3:
        # Boundary has shape [3, 2] where 3 is dimensions (x,y,z) and 2 is (min, max)
        lower_bound = boundary[:, 0]  
        upper_bound = boundary[:, 1]  
        box_size = upper_bound - lower_bound
        
        # Normalize recent position to [0, 1] range within the box
        normalized_position = (recent_position - lower_bound) / box_size
    else:
        print(f"Unexpected boundary shape: {boundary.shape}")
        raise ValueError("Boundary shape is not as expected.")

    if particle_type is None:
        # Flatten velocity sequence to create features
        # normal_velocity_seq has shape [num_particles, window_size-1, 3]
        flat_velocity = normal_velocity_seq.reshape(normal_velocity_seq.size(0), -1)
        
        if "temp_mean" in metadata and "temp_std" in metadata:
            temp_mean = torch.tensor(metadata["temp_mean"], dtype=torch.float32)
            temp_std = torch.tensor(metadata["temp_std"], dtype=torch.float32)
            normal_temp_seq = (temperature_history - temp_mean) / torch.sqrt(temp_std**2 + noise_std**2)
        else:
            print("Temperature metadata not found. Using raw temperature.")
            normal_temp_seq = temperature_history
        
        flat_temperature = normal_temp_seq.reshape(normal_temp_seq.size(0), -1)
        print(f"flat_temperature shape: {flat_temperature.shape}")
        
        node_features = torch.cat((flat_velocity, flat_temperature, normalized_position), dim=-1)
        
        print(f"flat_velocity shape: {flat_velocity.shape}")  # should be [num_particles, (window_size-1)*3]
        print(f"normalized_position shape: {normalized_position.shape}")  # should be [num_particles, 3]
        print(f"node_features shape: {node_features.shape}")  # should be [num_particles, (window_size-1)*3 + (window_size-1)*1 + 3]
    
    else:
        node_features = particle_type.float32
        
        
    # Edge-level features
    dim = recent_position.size(-1)
    senders = edge_index[0]
    receivers = edge_index[1]
    
    assert torch.max(senders) < len(recent_position), f"Max sender index {torch.max(senders)} >= {len(recent_position)}"
    assert torch.max(receivers) < len(recent_position), f"Max receiver index {torch.max(receivers)} >= {len(recent_position)}"

    edge_displacement = recent_position[senders] - recent_position[receivers]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
    
    acceleration = None
    temperature_change = None

    # Ground truth for training (acceleration)
    if target_position is not None:
        print(f"Target position original shape: {target_position.shape}")
        
        if target_position.dim() == 3:  # [1, num_particles, 3]
            target_position = target_position.permute(1, 0, 2)  # -> [num_particles, 1, 3]
            target_position = target_position.squeeze(1)  # -> [num_particles, 3]
        elif target_position.dim() == 2 and target_position.shape[0] != recent_position.shape[0]:
            target_position = target_position.reshape(-1, 3)
        
        print(f"Target position reshaped: {target_position.shape}")
        print(f"Recent position shape: {recent_position.shape}")
        
        last_velocity = velocity_seq[:, -1]
        next_velocity = target_position - recent_position
        
        # Apply noise correction
        if position_noise.dim() > 2:
            noise_term = position_noise[:, -1]
        else:
            noise_term = position_noise[-1]
        
        print(f"Noise term shape: {noise_term.shape}")
        next_velocity = next_velocity + noise_term
        acceleration = next_velocity - last_velocity
        
        acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
        acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
        acceleration = (acceleration - acc_mean) / torch.sqrt(acc_std**2 + noise_std**2)
    
    if target_temperature is not None:
        if target_temperature.dim() == 3:
            target_temperature = target_temperature.squeeze(1)
        
        print(f"Target temperature reshaped: {target_temperature.shape}")
        print(f"Recent temperature shape: {recent_temperature.shape}")
        
        temperature_change = target_temperature - recent_temperature
        
        if "temp_mean" in metadata and "temp_std" in metadata:
            temp_mean = torch.tensor(metadata["temp_mean"], dtype=torch.float32)
            temp_std = torch.tensor(metadata["temp_std"], dtype=torch.float32)
            temperature_change = (temperature_change - temp_mean) / torch.sqrt(temp_std**2 + noise_std**2)

    # Build a PyG Data object
    from torch_geometric.data import Data
    graph = Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=torch.cat((edge_displacement, edge_distance), dim=-1),
        y_acc=acceleration.float() if acceleration is not None else None,
        y_temp=temperature_change.float() if temperature_change is not None else None,
        pos=recent_position
    )
    return graph
