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
    
    print(f"Multi-simulation sequence-level split:")
    print(f"  Found {len(sim_files)} simulation files")
    
    full_dataset = SequenceDataset(
        paths=sim_files, 
        window_size=window_size,
        metadata=metadata,
        augment=False,  
        multi_simulation=True
    )
    
    total_sequences = len(full_dataset)
    
    all_indices = np.arange(total_sequences)
    np.random.shuffle(all_indices)
    
    # Split at sequence level, not simulation level
    split_idx = int(total_sequences * (1 - val_split))
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"  Total sequences across all simulations: {total_sequences}")
    print(f"  Training sequences: {len(train_indices)} ({len(train_indices)/total_sequences*100:.1f}%)")
    print(f"  Validation sequences: {len(val_indices)} ({len(val_indices)/total_sequences*100:.1f}%)")
    
    train_dataset = SequenceDataset(
        paths=sim_files,
        window_size=window_size,
        metadata=metadata,
        augment=augment,  # Enable augmentation for training
        augment_prob=augment_prob,
        start_indices=train_indices.tolist(), 
        multi_simulation=True
    )
    
    val_dataset = SequenceDataset(
        paths=sim_files,
        window_size=window_size,
        metadata=metadata,
        augment=False,  # Disable augmentation for validation
        start_indices=val_indices.tolist(),  
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