import numpy as np
import h5py
from dataloader import SequenceDataset

def get_train_val_datasets(data_path, window_size, metadata, val_split=0.2, augment=False, augment_prob=0.1, seed=42, multi_simulation=False):
    np.random.seed(seed)
    
    if multi_simulation:
        return _get_multi_sim_datasets(
            data_path, window_size, metadata, val_split, augment, augment_prob, seed
        )
    else:
        return _get_single_sim_datasets(
            data_path, window_size, metadata, val_split, augment, augment_prob, seed
        )

def _get_single_sim_datasets(data_path, window_size, metadata, val_split, augment, augment_prob, seed):
    with h5py.File(data_path, "r") as f:
        field_name = list(f.keys())[1]  
        num_snapshots = f[field_name].shape[0]
    
    # Calculate sequences and split
    num_sequences = num_snapshots - (window_size + 1) + 1
    all_indices = np.arange(num_sequences)
    np.random.shuffle(all_indices)
    
    split_idx = int(num_sequences * (1 - val_split))
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"Single simulation split: {len(train_indices)} training, {len(val_indices)} validation samples")
    
    train_dataset = SequenceDataset(
        paths=[data_path],
        window_size=window_size,
        metadata=metadata,
        augment=augment,  
        augment_prob=augment_prob,
        start_indices=train_indices.tolist(),
        multi_simulation=False
    )
    
    val_dataset = SequenceDataset(
        paths=[data_path],
        window_size=window_size,
        metadata=metadata,
        augment=False, 
        start_indices=val_indices.tolist(),
        multi_simulation=False
    )
    return train_dataset, val_dataset

def _get_multi_sim_datasets(data_path, window_size, metadata, val_split, augment, augment_prob, seed):
    import os
    from glob import glob
    
    if os.path.isdir(data_path):
        sim_files = sorted(glob(os.path.join(data_path, "*.hdf5")))
    elif isinstance(data_path, list):
        sim_files = data_path
    else:
        sim_files = [data_path]
    
    if not sim_files:
        raise FileNotFoundError(f"No simulation files found in {data_path}")
    
    # Split simulations between train and validation
    num_sims = len(sim_files)
    np.random.shuffle(sim_files)
    
    split_idx = int(num_sims * (1 - val_split))
    if split_idx == num_sims:  # If val_split is very small, ensure at least one validation sim
        split_idx = num_sims - 1
    if split_idx == 0:  # If val_split is very large, ensure at least one training sim
        split_idx = 1
        
    train_files = sim_files[:split_idx]
    val_files = sim_files[split_idx:]
    
    print(f"Multi-simulation split:")
    print(f"  Training simulations: {len(train_files)}")
    print(f"  Validation simulations: {len(val_files)}")
    
    # Calculate approximate number of samples
    with h5py.File(sim_files[0], "r") as f:
        field_name = list(f.keys())[1]
        num_snapshots = f[field_name].shape[0]
    
    sequences_per_sim = num_snapshots - window_size
    train_samples = len(train_files) * sequences_per_sim
    val_samples = len(val_files) * sequences_per_sim
    
    print(f"  Training samples: ~{train_samples}")
    print(f"  Validation samples: ~{val_samples}")
    
    train_dataset = SequenceDataset(
        paths=train_files,
        window_size=window_size,
        metadata=metadata,
        augment=augment,  
        augment_prob=augment_prob,
        multi_simulation=True
    )
    
    val_dataset = SequenceDataset(
        paths=val_files,
        window_size=window_size,
        metadata=metadata,
        augment=False,
        multi_simulation=True
    )
    return train_dataset, val_dataset

def get_simulation_based_split(simulation_dir, window_size, metadata, val_split=0.2, augment=False, augment_prob=0.1, seed=42):
    return get_train_val_datasets(
        data_path=simulation_dir,
        window_size=window_size,
        metadata=metadata,
        val_split=val_split,
        augment=augment,
        augment_prob=augment_prob,
        seed=seed,
        multi_simulation=True
    )