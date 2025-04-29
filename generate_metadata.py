# generate_metadata.py

import h5py
import numpy as np
import json
import argparse

def generate_metadata(dataset_path, output_path):
    """Generate metadata from HDF5 dataset and save to JSON."""
    with h5py.File(dataset_path, "r") as f:
        velocities = f['Velocities'][:]
        accelerations = f['HydroAcceleration'][:]
        coordinates = f['Coordinates'][:]
        internal_energy = f['InternalEnergy'][:] 

        temp_mean = np.mean(internal_energy, axis=(0, 1))
        temp_std = np.std(internal_energy, axis=(0, 1))
        
        vel_mean = np.mean(velocities, axis=(0, 1))
        vel_std = np.std(velocities, axis=(0, 1))
        
        acc_mean = np.mean(accelerations, axis=(0, 1))
        acc_std = np.std(accelerations, axis=(0, 1))
        
        # Manually set the box size 
        # min_coords = np.min(coordinates, axis=(0, 1))
        # max_coords = np.max(coordinates, axis=(0, 1))
        bounds = np.stack([min_coords, max_coords], axis=1)
        box_size = 2

        metadata = {
            "temp_mean": temp_mean.tolist(),
            "temp_std": temp_std.tolist(),
            "vel_mean": vel_mean.tolist(),
            "vel_std": vel_std.tolist(),
            "acc_mean": acc_mean.tolist(),
            "acc_std": acc_std.tolist(),
            "bounds": bounds.tolist(),
            "box_size": box_size.tolist()
        }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Metadata saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate metadata for dataset')
    parser.add_argument('--dataset', type=str, required=True, help='Path to HDF5 dataset')
    parser.add_argument('--output', type=str, default='metadata.json', help='Output path for metadata.json')
    
    args = parser.parse_args()
    generate_metadata(args.dataset, args.output)