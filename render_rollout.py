import os
import torch
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

from graph_network import EncodeProcessDecode
from data_utils import preprocess

def load_model(model_path, args):
    model = EncodeProcessDecode(
        latent_size=args.latent_size,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
        num_message_passing_steps=args.num_message_passing_steps,
        output_size=args.output_size
    )
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    return model

def rollout(model, data, metadata, noise_std, dt, box_size, window_size=6):
    device = next(model.parameters()).device
    model.eval()
    
    with torch.no_grad():
        total_time = data["Coordinates"].size(0)
        position_traj = data["Coordinates"][:window_size].permute(1, 0, 2).float() # -> [num_particles, window_size, 3]
        # Handle temperature data - ensure it's 3D before permuting
        temp_data = data["InternalEnergy"][:window_size]
        if temp_data.dim() == 2:
            temp_data = temp_data.unsqueeze(-1)  # Add feature dimension: [time_steps, num_particles, 1]
        temp_traj = temp_data.permute(1, 0, 2).float() # -> [num_particles, window_size, 1]

        for time in range(total_time - window_size):
            # Build a graph with no noise for rollout
            input_positions = position_traj[:, -window_size:].permute(1, 0, 2) # -> [window_size, num_particles, 3]
            input_temps = temp_traj[:, -window_size:].permute(1, 0, 2) # -> [window_size, num_particles, 1]
            
            graph = preprocess(
                position_seq=input_positions, 
                temperature_seq=input_temps,
                metadata=metadata, 
                noise_std=0.0, 
                num_neighbors=16, 
                box_size=box_size,
                dt=dt
            )
            graph = graph.to(device)

            # Predict acceleration and temperature rate
            predictions = model(graph)
            acc_pred = predictions['acceleration'].cpu()
            temp_rate_pred = predictions['temp_rate'].cpu()

            # Un-normalize acceleration
            acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
            acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
            
            acc_pred = acc_pred * torch.sqrt(acc_std**2 + noise_std**2) + acc_mean
            
            # Un-normalize temperature rate
            temp_rate_std = torch.tensor(metadata["temp_rate_std"], dtype=torch.float32)
            temp_rate_mean = torch.tensor(metadata["temp_rate_mean"], dtype=torch.float32)
            temp_rate_pred = temp_rate_pred * torch.sqrt(temp_rate_std**2 + noise_std**2) + temp_rate_mean
            
            # Obtain the recent values of position, velocity, temperature
            recent_position = position_traj[:, -1]
            recent_velocity = (recent_position - position_traj[:, -2]) / dt
            recent_temp = temp_traj[:, -1]
            
            # Integrate to new values, keeping in mind periodicity for position update
            new_velocity = recent_velocity + acc_pred * dt
            new_position = recent_position + new_velocity * dt
            
            # Perform modulo by box_size to keep particles within the box
            new_position = torch.remainder(new_position, box_size)
            new_temp = recent_temp + temp_rate_pred * dt
            
            position_traj = torch.cat((position_traj, new_position.unsqueeze(1)), dim=1) # -> [num_particles, time_steps + 1, 3]
            temp_traj = torch.cat((temp_traj, new_temp.unsqueeze(1)), dim=1) # -> [num_particles, time_steps + 1, 1]
            
        return {
            "Coordinates": position_traj.permute(1, 0, 2), # -> [total_time_steps, num_particles, 3]
            "InternalEnergy": temp_traj.permute(1, 0, 2) # -> [total_time_steps, num_particles, 1]
        }   

def calculate_errors(rollout_data, ground_truth):
    pred_coords = rollout_data["Coordinates"][:]
    true_coords = ground_truth["Coordinates"][:]
    
    # Calculate position errors
    pos_errors = []
    for t in range(len(pred_coords)):
        if t >= len(true_coords):
            break
        mse = torch.mean((pred_coords[t] - true_coords[t])**2).item()
        pos_errors.append(mse)
    
    # Calculate temperature errors
    temp_errors = []
    pred_temps = rollout_data["InternalEnergy"][:].squeeze()
    true_temps = ground_truth["InternalEnergy"][:].squeeze()
        
    for t in range(len(pred_temps)):
        if t >= len(true_temps):
            break
        mse = torch.mean((pred_temps[t] - true_temps[t])**2).item()
        temp_errors.append(mse)
    
    return {
        "position_errors": pos_errors,
        "temperature_errors": temp_errors,
        "mean_position_error": np.mean(pos_errors) if pos_errors else None,
        "mean_temperature_error": np.mean(temp_errors) if temp_errors else None
    }

def plot_errors(errors, output_path, window_size):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot position and temperature errors
    ax.plot(errors["position_errors"], 'b-', linewidth=2, label='Position MSE')
    ax.plot(errors["temperature_errors"], 'r-', linewidth=2, label='Temperature MSE')
    
    ax.set_title('Rollout Error', fontsize=14)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    plt.axvline(x=window_size, color='r', linestyle='--', linewidth=2, label='Rollout start')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Error plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Perform model rollout')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata')
    parser.add_argument('--output_dir', type=str, default='rollout_results', help='Output directory for results')
    parser.add_argument('--window_size', type=int, default=5, help='Input window size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Noise standard deviation')
    parser.add_argument('--latent_size', type=int, default=128, help='Model latent size')
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='Model MLP hidden size')
    parser.add_argument('--mlp_num_hidden_layers', type=int, default=2, help='Model MLP layers')
    parser.add_argument('--num_message_passing_steps', type=int, default=10, help='Model message passing steps')
    parser.add_argument('--output_size', type=int, default=3, help='Model output size (3 for 3D)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading metadata from {args.metadata_path}")
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get dt and box_size from metadata
    dt = metadata['dt']
    box_size = metadata['box_size']
    
    print(f"Using time step (dt): {dt}")
    print(f"Using box size: {box_size}")
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args)
    
    print(f"Loading test data from {args.test_data}")
    with h5py.File(args.test_data, 'r') as f:
        ground_truth = {
            "Coordinates": torch.tensor(f["Coordinates"][:], dtype=torch.float32),
            "InternalEnergy": torch.tensor(f["InternalEnergy"][:], dtype=torch.float32)
        }
    
    print("Performing rollout")
    rollout_data = rollout(
        model=model,
        data=ground_truth,
        metadata=metadata,
        noise_std=args.noise_std,
        dt=dt,
        box_size=box_size, 
        window_size=args.window_size
    )
    
    print("Calculating errors")
    errors = calculate_errors(
        rollout_data=rollout_data,
        ground_truth=ground_truth
    )
    
    print("Plotting errors")
    plot_errors(
        errors=errors,
        output_path=os.path.join(args.output_dir, "errors.png"),
        window_size=args.window_size
    )
    
    coords_path = os.path.join(args.output_dir, "rollout_coordinates.npy")
    np.save(coords_path, rollout_data["Coordinates"].detach().numpy())
    
    temps_path = os.path.join(args.output_dir, "rollout_temperatures.npy")
    np.save(temps_path, rollout_data["InternalEnergy"].detach().numpy())
    
    print(f"Rollout coordinates saved to {coords_path}")
    print(f"Rollout temperatures saved to {temps_path}")
    
    # Create a summary file with error statistics
    with open(os.path.join(args.output_dir, 'rollout_summary.txt'), 'w') as f:
        f.write("Rollout Summary\n")
        f.write("==============\n\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test data: {args.test_data}\n")
        f.write(f"Number of particles: {rollout_data['Coordinates'].shape[1]}\n")
        f.write(f"Window size: {args.window_size}\n")
        f.write(f"Time steps simulated: {rollout_data['Coordinates'].shape[0] - args.window_size}\n")
        f.write(f"Time step (dt): {dt}\n")
        f.write(f"Box size: {box_size}\n\n")
        
        if errors['mean_position_error'] is not None:
            f.write(f"Mean position MSE: {errors['mean_position_error']:.6e}\n")
        
        if errors['mean_temperature_error'] is not None:
            f.write(f"Mean temperature MSE: {errors['mean_temperature_error']:.6e}\n")
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()