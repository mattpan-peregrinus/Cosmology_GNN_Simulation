import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class SequenceDataset(Dataset):
    def __init__(
        self,
        paths,
        window_size,
        metadata, 
        augment,
        augment_prob,  
        start_indices=None,  
        **kwargs,
    ):
        if isinstance(paths, str):
            if os.path.isdir(paths):
                file_lists = sorted(glob(os.path.join(paths, "*.hdf5")))
                if not file_lists:
                    file_lists = sorted(glob(os.path.join(paths, "*.h5")))
                if not file_lists:
                    raise FileNotFoundError(f"No HDF5 files found in {paths}")
            else:
                file_lists = [paths]
        elif isinstance(paths, list):
            file_lists = paths
        else:
            raise ValueError("paths must be a directory, file, or list of files")

        self.file_lists = file_lists
        self.nfiles = len(file_lists)
        print(f"Initializing dataset with {self.nfiles} simulation file(s)")

        if self.nfiles == 0:
            raise FileNotFoundError("No files found")

        with h5py.File(self.file_lists[0], "r") as f:
            self.field_names = [field_name for field_name in f.keys() if f[field_name].ndim > 0]  
            self.num_snapshots = f[self.field_names[0]].shape[0]
            self.num_particles = f[self.field_names[0]].shape[1]
            self.ndims = []
            for field_name in self.field_names:
                data = f[field_name]
                if data.ndim == 2:  
                    self.ndims.append(1)
                else:  # [time, particles, features]
                    self.ndims.append(data.shape[-1])

        if self.nfiles > 1:
            for i, file_path in enumerate(self.file_lists[1:], 1):
                with h5py.File(file_path, "r") as f:
                    num_snapshots = f[self.field_names[0]].shape[0]
                    num_particles = f[self.field_names[0]].shape[1]
                    if num_snapshots != self.num_snapshots:
                        raise ValueError(f"File {file_path} has {num_snapshots} snapshots, "
                                       f"expected {self.num_snapshots}")
                    if num_particles != self.num_particles:
                        raise ValueError(f"File {file_path} has {num_particles} particles, "
                                       f"expected {self.num_particles}")

        self.metadata = metadata
        self.dt = metadata["dt"]
        self.box_size = metadata["box_size"]
        self.augment = augment  
        self.augment_prob = augment_prob
        self.window_size = window_size
        # Assertion 1: Check if the number of snapshots is larger than the window size
        assert self.num_snapshots >= self.window_size + 1, \
            f"num_snapshots ({self.num_snapshots}) must be larger than window_size + 1 ({self.window_size + 1})"
        self.num_sequences_per_sim = self.num_snapshots - self.window_size
        if start_indices is not None:
            self.start_indices = start_indices
            self.num_sequences = len(self.start_indices)
            # Validate that start_indices are within bounds
            max_possible_sequences = self.nfiles * self.num_sequences_per_sim
            max_index = max(self.start_indices) if self.start_indices else 0
            assert max_index < max_possible_sequences, \
                f"Invalid start index: {max_index} >= {max_possible_sequences}"
        else:
            self.start_indices = None  
            self.num_sequences = self.nfiles * self.num_sequences_per_sim
        self.num_samples = len(self.start_indices) if self.start_indices else (self.nfiles * self.num_sequences_per_sim)
        print(f"Dataset summary:")
        print(f"  - Number of simulations: {self.nfiles}")
        print(f"  - Snapshots per simulation: {self.num_snapshots}")
        print(f"  - Particles per snapshot: {self.num_particles}")
        print(f"  - Window size: {self.window_size}")
        print(f"  - Sequences per simulation: {self.num_sequences}")
        print(f"  - Total samples: {self.num_samples}")
        # Caching for multi-simulation mode
        self._cached_sim_idx = None
        self._cached_sim_data = None
        self.is_read_once = np.full(self.nfiles, False)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._get_multi_sim_item(idx)

    def _get_multi_sim_item(self, idx):
        if self.start_indices is not None:
            global_seq_idx = self.start_indices[idx]
            sim_idx, seq_idx = divmod(global_seq_idx, self.num_sequences_per_sim)
        else:
            sim_idx, seq_idx = divmod(idx, self.num_sequences_per_sim)
        
        start_idx = seq_idx
        end_idx = start_idx + self.window_size
        
        if sim_idx != self._cached_sim_idx:
            self._load_simulation(sim_idx)
            
        in_fields = {}
        tgt_fields = {}
        
        for field_name in self.field_names:
            in_fields[field_name] = self._cached_sim_data[field_name][start_idx:end_idx].astype(np.float32)
            tgt_fields[field_name] = self._cached_sim_data[field_name][end_idx:end_idx+1].astype(np.float32)
            
            if field_name == 'InternalEnergy':
                if len(in_fields[field_name].shape) == 2:
                    in_fields[field_name] = in_fields[field_name][..., np.newaxis]
                if len(tgt_fields[field_name].shape) == 2:
                    tgt_fields[field_name] = tgt_fields[field_name][..., np.newaxis]

        return self._process_fields(in_fields, tgt_fields)

    def _process_fields(self, in_fields, tgt_fields):
        in_fields = {key: torch.from_numpy(field).float() for key, field in in_fields.items()}
        tgt_fields = {key: torch.from_numpy(field).float() for key, field in tgt_fields.items()}
        
        # Apply augmentation if enabled 
        if self.augment:

            # Random permutation of xyz axis
            if np.random.random() < self.augment_prob:
                perm_idx = torch.randperm(3)
                for i, (key, field) in enumerate(in_fields.items()):
                    ndim = self.ndims[i] if i < len(self.ndims) else field.shape[-1]  
                    if ndim >= 2 and field.shape[-1] == 3:
                        in_fields[key] = field[..., perm_idx]
                for i, (key, field) in enumerate(tgt_fields.items()):
                    ndim = self.ndims[i] if i < len(self.ndims) else field.shape[-1]  
                    if ndim >= 2 and field.shape[-1] == 3:
                        tgt_fields[key] = field[..., perm_idx]

        return {
            "input": {
                **in_fields,
                "box_size": torch.tensor([self.box_size], dtype=torch.float32),
                "dt": torch.tensor([self.dt], dtype=torch.float32)
            },
            "target": tgt_fields,
        }

    def _load_simulation(self, sim_idx):
        file_path = self.file_lists[sim_idx]
        
        self._cached_sim_data = {}
        
        with h5py.File(file_path, "r") as f:
            for field_name in self.field_names:
                self._cached_sim_data[field_name] = f[field_name][:]
                
        self._cached_sim_idx = sim_idx