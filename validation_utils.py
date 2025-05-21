import numpy as np
import h5py
from dataloader import SequenceDataset

def get_train_val_datasets(data_path, window_size, val_split=0.2, augment=False, augment_prob=0.1, seed=42):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    with h5py.File(data_path, "r") as f:
        field_name = list(f.keys())[0]  
        num_snapshots = f[field_name].shape[0]
    
    # Calculate how many sequences we can make
    num_sequences = num_snapshots - (window_size + 1) + 1
    
    # Create indices and shuffle them
    all_indices = np.arange(num_sequences)
    np.random.shuffle(all_indices)
    
    # Split indices
    split_idx = int(num_sequences * (1 - val_split))
    train_indices = all_indices[:split_idx]
    val_indices = all_indices[split_idx:]
    
    print(f"Created train-val split: {len(train_indices)} training samples, {len(val_indices)} validation samples")
    
    # Create datasets with custom start indices
    train_dataset = SequenceDataset(
        paths=[data_path],
        window_size=window_size,
        metadata=metadata,
        augment=augment,  
        augment_prob=augment_prob,
        start_indices=train_indices.tolist()
    )
    
    val_dataset = SequenceDataset(
        paths=[data_path],
        window_size=window_size,
        metadata=metadata,
        augment=False, 
        start_indices=val_indices.tolist()
    )
    
    return train_dataset, val_dataset