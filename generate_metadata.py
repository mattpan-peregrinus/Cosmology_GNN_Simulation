import h5py
import numpy as np
import json
import argparse

def generate_metadata(dataset_path, output_path):
    with h5py.File(dataset_path, "r") as f:
        velocities = f['Velocities'][:]
        accelerations = f['HydroAcceleration'][:]
        coordinates = f['Coordinates'][:]
        internal_energy = f['InternalEnergy'][:]
        box_size = float(f['BoxSize'][...])
        dt = float(f['TimeStep'][...])
        
        temp_mean = np.mean(internal_energy, axis=(0, 1))
        temp_std = np.std(internal_energy, axis=(0, 1))

        temp_rate = (internal_energy[1:] - internal_energy[:-1]) / dt
        temp_rate_mean = np.mean(temp_rate, axis=(0, 1))
        temp_rate_std = np.std(temp_rate, axis=(0, 1))
        
        vel_mean = np.mean(velocities, axis=(0, 1))
        vel_std = np.std(velocities, axis=(0, 1))
        
        acc_mean = np.mean(accelerations, axis=(0, 1))
        acc_std = np.std(accelerations, axis=(0, 1))
        
        metadata = {
            "temp_mean": temp_mean.tolist(),
            "temp_std": temp_std.tolist(),
            "temp_rate_mean": temp_rate_mean.tolist(),
            "temp_rate_std": temp_rate_std.tolist(),
            "vel_mean": vel_mean.tolist(),
            "vel_std": vel_std.tolist(),
            "acc_mean": acc_mean.tolist(),
            "acc_std": acc_std.tolist(),
            "box_size": box_size,
            "dt": dt
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