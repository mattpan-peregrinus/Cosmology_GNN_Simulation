import os
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from mpl_toolkits.mplot3d import Axes3D

from graph_network import EncodeProcessDecode
from data_utils import preprocess
from config import get_config

def load_model(model_path, args):
    """Load a trained model."""
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

def perform_rollout(model, initial_data, metadata, window_size, num_steps, device, num_neighbors=16, noise_std=0.0):
    """
    Perform a rollout starting from initial conditions.
    """
    model.eval()
    
    dt = metadata.get('dt', 1.0) 
    box_size = metadata.get('box_size')
    if isinstance(box_size, list) and box_size:
        box_size = float(box_size[0])
    
    # Initialize with the known data
    coords_seq = initial_data["Coordinates"][:window_size].float()  # [window_size, num_particles, 3]
    temp_seq = initial_data["InternalEnergy"][:window_size].float()  # [window_size, num_particles, 1]
    
    # Check and fix temperature dimension if needed
    if len(temp_seq.shape) == 2:
        temp_seq = temp_seq.unsqueeze(-1)
    
    # Convert to the shape needed for history
    particle_history_coords = coords_seq.permute(1, 0, 2)  # [num_particles, window_size, 3]
    particle_history_temp = temp_seq.permute(1, 0, 2)  # [num_particles, window_size, 1]
    
    num_particles = particle_history_coords.shape[0]
    
    # Arrays to store results
    rolled_coords = torch.zeros((num_steps + window_size, num_particles, 3), dtype=torch.float32)
    rolled_temps = torch.zeros((num_steps + window_size, num_particles, 1), dtype=torch.float32)
    
    # Copy initial conditions
    rolled_coords[:window_size] = coords_seq
    rolled_temps[:window_size] = temp_seq
    
    # Perform rollout
    for step in tqdm(range(num_steps)):
        current_idx = step + window_size
        
        # Prepare input data for the current step
        window_coords = rolled_coords[current_idx-window_size:current_idx]
        window_temps = rolled_temps[current_idx-window_size:current_idx]
        
        # Create graph
        graph = preprocess(
            position_seq=window_coords,
            target_position=None,  # No target for prediction
            metadata=metadata,
            noise_std=noise_std,  # No noise for evaluation
            num_neighbors=num_neighbors,
            temperature_seq=window_temps,
            dt = dt,
            box_size = box_size
        )
        graph = graph.to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model(graph)
            acc_pred = predictions['acceleration'].cpu()
            temp_pred = predictions['temperature'].cpu()
        
        # Un-normalize acceleration
        acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
        acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
            
        acc_pred = acc_pred * torch.sqrt(acc_std**2 + noise_std**2) + acc_mean
        
        # Un-normalize temperature change
        if "temp_std" in metadata and "temp_mean" in metadata:
            temp_std = torch.tensor(metadata["temp_std"], dtype=torch.float32)
            temp_mean = torch.tensor(metadata["temp_mean"], dtype=torch.float32)
                
            temp_pred = temp_pred * torch.sqrt(temp_std**2 + noise_std**2) + temp_mean
        
        # Update position: get recent position and velocity
        recent_position = window_coords[-1]  # [num_particles, 3]
        recent_velocity = (recent_position - window_coords[-2]) / dt  # Scale by dt
        
        # Compute new velocity and position
        new_velocity = recent_velocity + acc_pred * dt
        new_position = recent_position + new_velocity * dt
        
        # Update temperature
        recent_temp = window_temps[-1]  # [num_particles, 1]
        new_temp = recent_temp + temp_pred * dt  # Scale by dt
        
        # Store results
        rolled_coords[current_idx] = new_position
        rolled_temps[current_idx] = new_temp
    
    return {
        "Coordinates": rolled_coords,
        "InternalEnergy": rolled_temps
    }

def evaluate_against_ground_truth(rollout_data, ground_truth, window_size):
    """
    Evaluate rolled out trajectories against ground truth if available.
    """
    # Extract predictions and ground truth
    pred_coords = rollout_data["Coordinates"][window_size:]
    pred_temps = rollout_data["InternalEnergy"][window_size:]
    
    # Limit comparison to available ground truth
    max_steps = min(len(pred_coords), len(ground_truth["Coordinates"]) - window_size)
    
    true_coords = ground_truth["Coordinates"][window_size:window_size+max_steps]
    true_temps = ground_truth["InternalEnergy"][window_size:window_size+max_steps]
    
    # Calculate errors
    coord_errors = []
    temp_errors = []
    
    for t in range(max_steps):
        # MSE for coordinates
        coord_mse = torch.mean((pred_coords[t] - true_coords[t])**2).item()
        coord_errors.append(coord_mse)
        
        # MSE for temperature if available
        if true_temps is not None:
            true_temp = true_temps[t]
            if len(true_temp.shape) == 1:
                true_temp = true_temp.unsqueeze(-1)
            
            pred_temp = pred_temps[t]
            if len(pred_temp.shape) == 1:
                pred_temp = pred_temp.unsqueeze(-1)
                
            temp_mse = torch.mean((pred_temp - true_temp)**2).item()
            temp_errors.append(temp_mse)
    
    return {
        "coord_errors": coord_errors,
        "temp_errors": temp_errors,
        "mean_coord_error": np.mean(coord_errors),
        "mean_temp_error": np.mean(temp_errors) if temp_errors else None
    }
    
def main():
    parser = argparse.ArgumentParser(description='Perform model rollout evaluation')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data (fullrun3.hdf5)')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata JSON')
    parser.add_argument('--output_dir', type=str, default='rollout_results', help='Directory to save results')
    
    # Optional arguments
    parser.add_argument('--window_size', type=int, default=5, help='Input window size (same as training)')
    parser.add_argument('--num_steps', type=int, default=500, help='Number of steps to roll out')
    parser.add_argument('--num_neighbors', type=int, default=16, help='Number of neighbors for graph')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--latent_size', type=int, default=128, help='Model latent size')
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='Model MLP hidden size')
    parser.add_argument('--mlp_num_hidden_layers', type=int, default=2, help='Model MLP layers')
    parser.add_argument('--num_message_passing_steps', type=int, default=10, help='Model message passing steps')
    parser.add_argument('--output_size', type=int, default=3, help='Model output size (3 for 3D)')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    
    # Import metadata
    import json
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = load_model(args.model_path, args)
    
    print(f"Loading test data from {args.test_data}")
    
    # Load test data
    with h5py.File(args.test_data, 'r') as f:
        # Load ground truth for comparison
        ground_truth = {
            "Coordinates": torch.tensor(f["Coordinates"][:], dtype=torch.float32),
        }
        
        if "InternalEnergy" in f:
            internal_energy = torch.tensor(f["InternalEnergy"][:], dtype=torch.float32)
            if len(internal_energy.shape) == 2:
                internal_energy = internal_energy.unsqueeze(-1)
            ground_truth["InternalEnergy"] = internal_energy
    
    print(f"Performing rollout for {args.num_steps} steps")
    
    # Perform rollout
    rollout_data = perform_rollout(
        model=model,
        initial_data=ground_truth,
        metadata=metadata,
        window_size=args.window_size,
        num_steps=args.num_steps,
        device=args.device,
        num_neighbors=args.num_neighbors
    )
    
    errors = evaluate_against_ground_truth(
        rollout_data=rollout_data,
        ground_truth=ground_truth,
        window_size=args.window_size
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save summary text file
    with open(os.path.join(args.output_dir, 'rollout_summary.txt'), 'w') as f:
        f.write(f"Rollout Summary\n")
        f.write(f"==============\n\n")
        f.write(f"Number of particles: {rollout_data['Coordinates'].shape[1]}\n")
        f.write(f"Window size: {args.window_size}\n")
        f.write(f"Prediction steps: {args.num_steps}\n\n")
        f.write(f"Mean position error: {errors['mean_coord_error']:.6e}\n")
        if errors['mean_temp_error'] is not None:
            f.write(f"Mean temperature error: {errors['mean_temp_error']:.6e}\n")
        
        # Also write the error at each time step
        f.write("\nPosition error at each time step:\n")
        for t, err in enumerate(errors['coord_errors']):
            f.write(f"Step {t}: {err:.6e}\n")
            
        if errors['temp_errors']:
            f.write("\nTemperature error at each time step:\n")
            for t, err in enumerate(errors['temp_errors']):
                f.write(f"Step {t}: {err:.6e}\n")
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()