import torch
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm
import os

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

def validate_one_step(model, data_path, metadata, window_size, device, num_neighbors=16, num_timesteps=10, noise_std=0.0):
    model.eval()
    position_errors = []
    temperature_errors = []
    tested_timesteps = []
    
    dt = metadata['dt']
    box_size = metadata['box_size']
    
    print(f"Loading data from {data_path}")
    with h5py.File(data_path, 'r') as f:
        total_frames = f["Coordinates"].shape[0]
        num_particles = f["Coordinates"].shape[1]
        max_start_idx = total_frames - window_size - 1 
        
        if max_start_idx < num_timesteps:
            print(f"Warning: Only {max_start_idx} valid timesteps available, reducing from {num_timesteps}")
            num_timesteps = max_start_idx
        
        start_indices = np.random.choice(max_start_idx, size=num_timesteps, replace=False)
        start_indices = sorted(start_indices)
        
        print(f"Testing {num_timesteps} timesteps: {start_indices}")
        print(f"Each test uses {window_size} input frames to predict the next frame")
        print(f"Data shape: {total_frames} timesteps, {num_particles} particles")
        
        for start_idx in tqdm(start_indices, desc="Testing timesteps"):
            # Handle position data
            coords_seq = torch.tensor(f["Coordinates"][start_idx:start_idx+window_size], dtype=torch.float32)
            next_coords = torch.tensor(f["Coordinates"][start_idx+window_size], dtype=torch.float32)
            
            # Handle temperature data
            temp_seq = torch.tensor(f["InternalEnergy"][start_idx:start_idx+window_size], dtype=torch.float32)
            next_temp = torch.tensor(f["InternalEnergy"][start_idx+window_size], dtype=torch.float32)
            
            # Ensure temperature data is 3D: [time_steps, num_particles, 1]
            if temp_seq.dim() == 2:
                temp_seq = temp_seq.unsqueeze(-1)  # Add feature dimension
            if next_temp.dim() == 1:
                next_temp = next_temp.unsqueeze(-1)  # Add feature dimension

            graph = preprocess(
                position_seq=coords_seq,
                temperature_seq=temp_seq,
                metadata=metadata,
                noise_std=noise_std,  
                num_neighbors=num_neighbors,
                box_size=box_size,  
                dt=dt  
            )
            graph = graph.to(device)
            
            # Predict
            with torch.no_grad():
                predictions = model(graph)
                acc_pred = predictions['acceleration'].cpu()
                temp_rate_pred = predictions['temp_rate'].cpu()
            
            # Un-normalize acceleration
            acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
            acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
            acc_pred = (acc_pred * acc_std) + acc_mean
            
            # Un-normalize temperature rate
            temp_rate_std = torch.tensor(metadata["temp_rate_std"], dtype=torch.float32)
            temp_rate_mean = torch.tensor(metadata["temp_rate_mean"], dtype=torch.float32)
            temp_rate_pred = (temp_rate_pred * temp_rate_std) + temp_rate_mean
            
            # Obtain the recent values of position, velocity, temperature
            recent_position = coords_seq[-1] 
            recent_velocity = (recent_position - coords_seq[-2]) / dt
            recent_temp = temp_seq[-1] 
            
            # Integrate to new values, keeping in mind periodicity for position update
            new_velocity = recent_velocity + acc_pred * dt
            new_position = recent_position + new_velocity * dt
            
            # Perform modulo by box_size to keep particles within the box
            new_position = torch.remainder(new_position, box_size)
            new_temp = recent_temp + temp_rate_pred * dt
            
            # Calculate position and temperature error (averaged over ALL particles)
            position_mse = torch.mean((new_position - next_coords) ** 2).item()
            position_errors.append(position_mse) 
            temperature_mse = torch.mean((new_temp - next_temp) ** 2).item()
            temperature_errors.append(temperature_mse)
            tested_timesteps.append(start_idx + window_size)
    
    # Compute average errors across all tested timesteps
    avg_position_error = np.mean(position_errors)
    avg_temperature_error = np.mean(temperature_errors) 
    
    return {
        "position_error": avg_position_error,
        "temperature_error": avg_temperature_error,
        "position_errors": position_errors,
        "temperature_errors": temperature_errors,
        "tested_timesteps": tested_timesteps
    }

def main():
    parser = argparse.ArgumentParser(description='Validate one-step predictions')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data (ex: fullrun3.hdf5)')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata JSON')
    parser.add_argument('--window_size', type=int, default=5, help='Input window size (same as training)')
    parser.add_argument('--num_neighbors', type=int, default=16, help='Number of neighbors for graph')
    parser.add_argument('--num_timesteps', type=int, default=10, help='Number of timesteps to validate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--latent_size', type=int, default=128, help='Model latent size')
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='Model MLP hidden size')
    parser.add_argument('--mlp_num_hidden_layers', type=int, default=2, help='Model MLP layers')
    parser.add_argument('--num_message_passing_steps', type=int, default=10, help='Model message passing steps')
    parser.add_argument('--output_size', type=int, default=3, help='Model output size (3 for 3D)')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    model = load_model(args.model_path, args)
    print(f"Validating one-step predictions on {args.num_timesteps} timesteps from {args.test_data}")
    
    # Validate one-step predictions
    results = validate_one_step(
        model=model,
        data_path=args.test_data,
        metadata=metadata,
        window_size=args.window_size,
        device=args.device,
        num_neighbors=args.num_neighbors,
        num_timesteps=args.num_timesteps
    )
    
    print("\n" + "="*50)
    print("ONE-STEP VALIDATION RESULTS")
    print("="*50)
    print(f"Number of timesteps tested: {len(results['position_errors'])}")
    print(f"Tested timesteps: {results['tested_timesteps']}")
    print(f"Average position MSE: {results['position_error']:.6e}")
    print(f"Average temperature MSE: {results['temperature_error']:.6e}")
    print(f"Position MSE std: {np.std(results['position_errors']):.6e}")
    print(f"Temperature MSE std: {np.std(results['temperature_errors']):.6e}")
    
    print(f"\nPer-timestep breakdown:")
    print(f"{'Timestep':<10} {'Position MSE':<15} {'Temperature MSE'}")
    print("-" * 40)
    for i, timestep in enumerate(results['tested_timesteps']):
        print(f"{timestep:<10} {results['position_errors'][i]:<15.6e} {results['temperature_errors'][i]:.6e}")

if __name__ == "__main__":
    main()