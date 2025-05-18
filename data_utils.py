import torch
import torch_geometric as pyg
from torch_geometric.nn import knn_graph
from torch_cluster import knn
from torch_geometric.data import Data
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


def preprocess(position_seq, temperature_seq, metadata, target_position=None, target_temperature=None, noise_std=0.0, num_neighbors=16):
    """Preprocess a trajectory and construct a PyG Data object with both position and temperature target."""
    # Determine box size and timestep
    box_size = metadata["box_size"]
    if isinstance(box_size, list):
        box_size = float(box_size[0])
    dt = metadata["dt"]
    if isinstance(dt, list):
        dt = float(dt[0])
    
    position_seq = position_seq.float()
    if target_position is not None:
        target_position = target_position.float()
    temperature_seq = temperature_seq.float()
    
    # Rearrange dimensions if needed
    position_seq = position_seq.permute(1, 0, 2)    
    if temperature_seq.shape[0] == position_seq.shape[1] and temperature_seq.shape[1] == position_seq.shape[0]:
        temperature_seq = temperature_seq.permute(1, 0, 2)    
    
    # Apply noise
    position_noise = generate_noise(position_seq, noise_std)
    position_seq = position_seq + position_noise

    # Compute velocities
    recent_position = position_seq[:, -1]
    # Get displacements first
    displacement_seq = position_seq[:, 1:] - position_seq[:, :-1]
    # Correct for boundary crossings
    displacement_seq[displacement_seq < -1*box_size/2] += box_size
    displacement_seq[displacement_seq > box_size/2] -= box_size
    # Divide corrected displacements by dt to get velocities
    velocity_seq = displacement_seq / dt

    # Get recent temperature
    recent_temperature = temperature_seq[:, -1] 
    
    if target_temperature is not None:
        target_temperature = target_temperature.float()
        if target_temperature.dim() == 3:  # [1, num_particles, 1]
            target_temperature = target_temperature.permute(1, 0, 2)  # -> [num_particles, 1, 1]
            target_temperature = target_temperature.squeeze(1)  # -> [num_particles, 1]
        elif target_temperature.dim() == 2 and target_temperature.shape[1] != 1:
            target_temperature = target_temperature.reshape(-1, 1)
        
        if target_temperature.shape != recent_temperature.shape:
            print(f"Shape mismatch: target_temp {target_temperature.shape} vs recent_temp {recent_temperature.shape}")
            if target_temperature.numel() == recent_temperature.numel():
                target_temperature = target_temperature.reshape(recent_temperature.shape)

    # Node-level features
    vel_mean = torch.tensor(metadata["vel_mean"], dtype=torch.float32)
    vel_std = torch.tensor(metadata["vel_std"], dtype=torch.float32)
    normal_velocity_seq = (velocity_seq - vel_mean) / torch.sqrt(vel_std**2 + noise_std**2)
    
    # Process temperature features
    if "temp_mean" in metadata and "temp_std" in metadata:
        temp_mean = torch.tensor(metadata["temp_mean"], dtype=torch.float32)
        temp_std = torch.tensor(metadata["temp_std"], dtype=torch.float32)
        normal_temp_seq = (temperature_seq - temp_mean) / torch.sqrt(temp_std**2 + noise_std**2)
    else:
        normal_temp_seq = temperature_seq
    
    # Flatten features for input to the model
    flat_velocity = normal_velocity_seq.reshape(normal_velocity_seq.size(0), -1)
    flat_temperature = normal_temp_seq.reshape(normal_temp_seq.size(0), -1)
        
    # Create node features: just velocity sequency and temperature sequence
    node_features = torch.cat((
            flat_velocity, 
            flat_temperature,
        ), dim=-1)
    
    # Construct the graph via k-nearest neighbors with periodicity
    extended_positions, mapping = extend_positions_torch(recent_position, box_size)
    edge_index = knn(extended_positions, recent_position, num_neighbors)
    edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index[0] = mapping[edge_index[0]]
    edge_index[1] = mapping[edge_index[1]]

    # Edge-level features
    senders = edge_index[0]
    receivers = edge_index[1]

    assert torch.max(senders) < len(recent_position), f"Max sender index {torch.max(senders)} >= {len(recent_position)}"
    assert torch.max(receivers) < len(recent_position), f"Max receiver index {torch.max(receivers)} >= {len(recent_position)}"

    # Compute displacement and distance, concatenate into the edge attribute
    edge_displacement = recent_position[senders] - recent_position[receivers]
    edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
    edge_attr = torch.cat((edge_displacement, edge_distance), dim=-1)
    
    acceleration = None
    temp_rate = None

    # Ground truth for training (acceleration)
    if target_position is not None:
        if target_position.dim() == 3:  # [1, num_particles, 3]
            target_position = target_position.permute(1, 0, 2)  # -> [num_particles, 1, 3]
            target_position = target_position.squeeze(1)  # -> [num_particles, 3]
        elif target_position.dim() == 2 and target_position.shape[0] != recent_position.shape[0]:
            target_position = target_position.reshape(-1, 3)
        
        last_velocity = velocity_seq[:, -1]  # Already scaled by dt

        # Compute the next velocity, taking into account periodicity
        next_displacement = target_position - recent_position
        next_displacement[next_displacement < -1*box_size/2] += box_size
        next_displacement[next_displacement > box_size/2] -= box_size
        next_velocity = next_displacement / dt
                                
        # Apply noise correction
        if position_noise.dim() > 2:
            noise_term = position_noise[:, -1]
        else:
            noise_term = position_noise[-1]
        
        next_velocity = next_velocity + noise_term/dt  # Scale noise by dt
        
        # Calculate acceleration (change in velocity over time)
        acceleration = (next_velocity - last_velocity) / dt  # Scale by dt again for acceleration
        
        # Normalize acceleration
        acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
        acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)

        acceleration = (acceleration - acc_mean) / torch.sqrt(acc_std**2 + noise_std**2)
        
    if target_temperature is not None:
        if target_temperature.dim() == 3:
            target_temperature = target_temperature.squeeze(1)
        
        # Calculate temperature change rate (per unit time)
        temp_rate = (target_temperature - recent_temperature) / dt
        
        if "temp_rate_mean" in metadata and "temp_rate_std" in metadata:
            # Scale temperature change statistics by dt
            temp_rate_mean = torch.tensor(metadata["temp_rate_mean"], dtype=torch.float32)
            temp_rate_std = torch.tensor(metadata["temp_rate_std"], dtype=torch.float32)
            temp_rate = (temp_rate - temp_rate_mean) / torch.sqrt(temp_rate_std**2 + noise_std**2)
                

    # Build a PyG Data object
    graph = Data(
        x=node_features.float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_acc=acceleration.float() if acceleration is not None else None,
        y_temp=temp_rate.float() if temp_rate is not None else None,
        pos=recent_position,
        dt=torch.tensor([dt], dtype=torch.float32),  
        box_size=torch.tensor([box_size], dtype=torch.float32) if box_size is not None else None  
    )
    return graph
