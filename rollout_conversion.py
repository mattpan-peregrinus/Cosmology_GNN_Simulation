import numpy as np
import h5py
import argparse
import json
import os

def convert_rollout_to_hdf5(rollout_dir, original_data_path, metadata_path, output_path, window_size=6):
    """
    Convert rollout results to HDF5 format matching the original data structure.
    
    Args:
        rollout_dir: Directory containing rollout results (.npy files)
        original_data_path: Path to original HDF5 file (for reference structure and initial conditions)
        metadata_path: Path to metadata JSON file
        output_path: Output path for the new HDF5 file
        window_size: Number of initial timesteps taken from original data
    """
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    dt = metadata['dt']
    box_size = metadata['box_size']
    
    # Load rollout results
    rollout_coords_path = os.path.join(rollout_dir, 'rollout_coordinates.npy')
    rollout_temps_path = os.path.join(rollout_dir, 'rollout_temperatures.npy')
    
    if not os.path.exists(rollout_coords_path) or not os.path.exists(rollout_temps_path):
        raise FileNotFoundError(f"Rollout files not found in {rollout_dir}")
    
    rollout_coords = np.load(rollout_coords_path)  # Shape: [total_time_steps, num_particles, 3]
    rollout_temps = np.load(rollout_temps_path)    # Shape: [total_time_steps, num_particles, 1]
    
    print(f"Loaded rollout coordinates: {rollout_coords.shape}")
    print(f"Loaded rollout temperatures: {rollout_temps.shape}")
    
    # Load original data to get initial conditions and reference structure
    with h5py.File(original_data_path, 'r') as f_orig:
        # Get the structure and some initial data
        orig_coords = f_orig['Coordinates'][:]
        orig_temps = f_orig['InternalEnergy'][:]
        orig_velocities = f_orig['Velocities'][:]
        
        # Get other fields that we need to copy
        orig_box_size = f_orig['BoxSize'][...]
        orig_timestep = f_orig['TimeStep'][...]
        
        print(f"Original data - Coords: {orig_coords.shape}, Temps: {orig_temps.shape}")
        
        # Verify dimensions match
        if rollout_coords.shape[1:] != orig_coords.shape[1:]:
            raise ValueError(f"Particle dimension mismatch: rollout {rollout_coords.shape[1:]} vs original {orig_coords.shape[1:]}")
    
    # Calculate velocities from rollout coordinates
    print("Calculating velocities from rollout coordinates...")
    rollout_velocities = np.zeros_like(rollout_coords)
    
    for t in range(1, rollout_coords.shape[0]):
        # Calculate displacement with periodic boundary conditions
        displacement = rollout_coords[t] - rollout_coords[t-1]
        
        # Handle periodic boundaries
        displacement[displacement > box_size/2] -= box_size
        displacement[displacement < -box_size/2] += box_size
        
        # Convert displacement to velocity
        rollout_velocities[t] = displacement / dt
    
    # For the first timestep, use the original velocity or calculate from initial positions
    if window_size > 1:
        # Use the velocity from the window_size-1 timestep from original data
        rollout_velocities[0] = orig_velocities[window_size-1]
    else:
        # Calculate initial velocity from first two positions
        displacement = rollout_coords[1] - rollout_coords[0]
        displacement[displacement > box_size/2] -= box_size
        displacement[displacement < -box_size/2] += box_size
        rollout_velocities[0] = displacement / dt
    
    # Calculate accelerations (HydroAcceleration field)
    print("Calculating accelerations...")
    rollout_accelerations = np.zeros_like(rollout_coords)
    
    for t in range(1, rollout_coords.shape[0]):
        # Calculate acceleration as change in velocity over time
        velocity_change = rollout_velocities[t] - rollout_velocities[t-1]
        rollout_accelerations[t] = velocity_change / dt
    
    # First timestep acceleration can be set to zero or copied from original
    rollout_accelerations[0] = rollout_accelerations[1]  # Use second timestep value
    
    # Create the HDF5 file
    print(f"Creating HDF5 file: {output_path}")
    with h5py.File(output_path, 'w') as f_out:
        # Create datasets with the same structure as original
        f_out.create_dataset('Coordinates', data=rollout_coords.astype(np.float32))
        f_out.create_dataset('InternalEnergy', data=rollout_temps.astype(np.float32))
        f_out.create_dataset('Velocities', data=rollout_velocities.astype(np.float32))
        f_out.create_dataset('HydroAcceleration', data=rollout_accelerations.astype(np.float32))
        
        # Copy scalar fields
        f_out.create_dataset('BoxSize', data=box_size)
        f_out.create_dataset('TimeStep', data=dt)
        
        print("HDF5 file created successfully with datasets:")
        for key in f_out.keys():
            if hasattr(f_out[key], 'shape'):
                print(f"  {key}: {f_out[key].shape} ({f_out[key].dtype})")
            else:
                print(f"  {key}: {f_out[key][...]} (scalar)")

def main():
    parser = argparse.ArgumentParser(description='Convert rollout results to HDF5 format')
    parser.add_argument('--rollout_dir', type=str, required=True, 
                       help='Directory containing rollout results')
    parser.add_argument('--original_data', type=str, required=True,
                       help='Path to original HDF5 data file')
    parser.add_argument('--metadata_path', type=str, required=True,
                       help='Path to metadata JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for converted HDF5 file')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Window size used in rollout (default: 5)')
    
    args = parser.parse_args()
    
    # Verify input files exist
    if not os.path.exists(args.rollout_dir):
        raise FileNotFoundError(f"Rollout directory not found: {args.rollout_dir}")
    
    if not os.path.exists(args.original_data):
        raise FileNotFoundError(f"Original data file not found: {args.original_data}")
    
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    convert_rollout_to_hdf5(
        rollout_dir=args.rollout_dir,
        original_data_path=args.original_data,
        metadata_path=args.metadata_path,
        output_path=args.output,
        window_size=args.window_size
    )
    
    print(f"Conversion complete! HDF5 file saved as: {args.output}")

if __name__ == "__main__":
    main()