import torch
import h5py
import json
import argparse
import numpy as np
from tqdm import tqdm

from graph_network import EncodeProcessDecode
from data_utils import preprocess

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

def validate_one_step(model, data_path, metadata, window_size, device, num_neighbors=16, num_examples=10, noise_std=0.0):
    """
    Validate one-step predictions over multiple examples.
    Returns the average error metrics.
    """
    model.eval()
    
    position_errors = []
    temperature_errors = []
    
    with h5py.File(data_path, 'r') as f:
        total_frames = f["Coordinates"].shape[0]
        max_start_idx = total_frames - window_size - 1 
        
        start_indices = np.random.choice(max_start_idx, size=min(num_examples, max_start_idx), replace=False)
        
        for start_idx in tqdm(start_indices, desc="Validating examples"):
            end_idx = start_idx + window_size + 1
            coords_seq = torch.tensor(f["Coordinates"][start_idx:start_idx+window_size], dtype=torch.float32)
            next_coords = torch.tensor(f["Coordinates"][start_idx+window_size], dtype=torch.float32)
            
            # Handle temperature data if available
            if "InternalEnergy" in f:
                temp_seq = torch.tensor(f["InternalEnergy"][start_idx:start_idx+window_size], dtype=torch.float32)
                next_temp = torch.tensor(f["InternalEnergy"][start_idx+window_size], dtype=torch.float32)
                
                # Ensure proper dimensions
                if len(temp_seq.shape) == 2:
                    temp_seq = temp_seq.unsqueeze(-1)
                if len(next_temp.shape) == 1:
                    next_temp = next_temp.unsqueeze(-1)
            else:
                temp_seq = None
                next_temp = None
            
            # Create graph
            graph = preprocess(
                position_seq=coords_seq,
                target_position=None,  
                metadata=metadata,
                noise_std=noise_std,  
                num_neighbors=num_neighbors,
                temperature_seq=temp_seq
            )
            graph = graph.to(device)
            
            # Predict
            with torch.no_grad():
                predictions = model(graph)
                acc_pred = predictions['acceleration'].cpu()
                temp_pred = predictions['temperature'].cpu() if temp_seq is not None else None
            
            # Un-normalize predictions
            acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
            acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
            acc_pred = acc_pred * torch.sqrt(acc_std**2 + noise_std**2) + acc_mean
            
            if temp_pred is not None and "temp_std" in metadata and "temp_mean" in metadata:
                temp_std = torch.tensor(metadata["temp_std"], dtype=torch.float32)
                temp_mean = torch.tensor(metadata["temp_mean"], dtype=torch.float32)
                temp_pred = temp_pred * torch.sqrt(temp_std**2 + noise_std**2) + temp_mean
            
            # Apply Euler integration
            recent_position = coords_seq[-1] 
            recent_velocity = recent_position - coords_seq[-2] 
            
            new_velocity = recent_velocity + acc_pred
            predicted_position = recent_position + new_velocity
            
            # Calculate position error
            position_mse = torch.mean((predicted_position - next_coords) ** 2).item()
            position_errors.append(position_mse)
            
            # Calculate temperature error if available
            if temp_pred is not None and next_temp is not None:
                recent_temp = temp_seq[-1] 
                predicted_temp = recent_temp + temp_pred
                
                temperature_mse = torch.mean((predicted_temp - next_temp) ** 2).item()
                temperature_errors.append(temperature_mse)
    
    # Compute average errors
    avg_position_error = np.mean(position_errors)
    avg_temperature_error = np.mean(temperature_errors) if temperature_errors else None
    
    return {
        "position_error": avg_position_error,
        "temperature_error": avg_temperature_error,
        "position_errors": position_errors,
        "temperature_errors": temperature_errors,
    }

def main():
    parser = argparse.ArgumentParser(description='Validate one-step predictions')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data (ex: fullrun3.hdf5)')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata JSON')
    
    # Optional arguments
    parser.add_argument('--window_size', type=int, default=5, help='Input window size (same as training)')
    parser.add_argument('--num_neighbors', type=int, default=16, help='Number of neighbors for graph')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to validate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--latent_size', type=int, default=128, help='Model latent size')
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='Model MLP hidden size')
    parser.add_argument('--mlp_num_hidden_layers', type=int, default=2, help='Model MLP layers')
    parser.add_argument('--num_message_passing_steps', type=int, default=10, help='Model message passing steps')
    parser.add_argument('--output_size', type=int, default=3, help='Model output size (3 for 3D)')
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}")
    
    # Import metadata
    with open(args.metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model = load_model(args.model_path, args)
    
    print(f"Validating one-step predictions on {args.num_examples} examples from {args.test_data}")
    
    # Validate one-step predictions
    results = validate_one_step(
        model=model,
        data_path=args.test_data,
        metadata=metadata,
        window_size=args.window_size,
        device=args.device,
        num_neighbors=args.num_neighbors,
        num_examples=args.num_examples
    )
    
    # Print results
    print("\nOne-Step Validation Results:")
    print(f"Average position MSE: {results['position_error']:.6e}")
    if results['temperature_error'] is not None:
        print(f"Average temperature MSE: {results['temperature_error']:.6e}")
    
    # Print per-example errors if requested
    print("\nPosition errors per example:")
    for i, err in enumerate(results['position_errors']):
        print(f"  Example {i}: {err:.6e}")
    
    if results['temperature_errors']:
        print("\nTemperature errors per example:")
        for i, err in enumerate(results['temperature_errors']):
            print(f"  Example {i}: {err:.6e}")

if __name__ == "__main__":
    main()