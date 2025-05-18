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
        norms=None,
        augment=False,
        augment_prob=0.1,
        start_indices=None,
        **kwargs,
    ):

        file_lists = paths
        self.file_lists = file_lists
        self.nfiles = len(file_lists)
        
        
        print(f"initializing with {self.nfiles} files")

        if self.nfiles == 0:
            raise FileNotFoundError("file not found for {}".format(paths))
        self.is_read_once = np.full(self.nfiles, False)

        with h5py.File(paths[0], "r") as f:
            self.field_names = [field_name for field_name in f.keys()]
            self.num_snapshots = f[self.field_names[0]].shape[0]
            self.num_particles = f[self.field_names[0]].shape[1]
            self.ndims = [f[field_name][:].shape[-1] for field_name in self.field_names]
            
            if "BoxSize" in f.attrs:
                self.box_size = f.attrs["BoxSize"]
            else:
                print("No BoxSize in dataset!!!")
                self.box_size = 2.0
                
            if "TimeStep" in f.attrs:
                self.dt = f.attrs["TimeStep"]
            else:
                print("No TimeStep in dataset!!!")
                self.dt = 1.0

        self.norms = norms
        self.augment = augment
        self.augment_prob = augment_prob
        self.window_size = window_size
        
        # Assertion 1: Check if the number of snapshots is larger than the window size
        assert self.num_snapshots >= self.window_size + 1, "num_snapshots must be larger than window_size"
        
        total_possible_sequences = self.num_snapshots - (self.window_size + 1) + 1
        if start_indices is not None:
            self.start_indices = start_indices
            self.num_sequences = len(self.start_indices)
            max_index = max(self.start_indices) if self.start_indices else 0
            assert max_index < total_possible_sequences, f"Invalid start index: {max_index} >= {total_possible_sequences}"
        else:
            self.start_indices = list(range(total_possible_sequences))
            self.num_sequences = total_possible_sequences
            
        self.num_samples = self.nfiles * self.num_sequences
        
        # Assertion 2: Check that number of sequences has to be equal to number of starting indices 
        assert len(self.start_indices) == self.num_sequences, "num_sequences must be equal to number of starting indices"
        
        # For debugging (summary of particles dataset)
        '''
        print(f"num_particles: {self.num_particles}")
        print(f"num_snapshots: {self.num_snapshots}")
        print(f"window_size: {self.window_size}")
        print(f"num_sequences: {self.num_sequences}")
        print(f"num_samples: {self.num_samples}")
        print(f"start indices = {self.start_indices}")
        print(f"start_indices length = {len(self.start_indices)}")
        '''
        

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        ifile, iseq = divmod(idx, self.num_sequences)
        start_idx = self.start_indices[iseq]
        end_idx = start_idx + self.window_size

        if not self.is_read_once[ifile]:
            self.is_read_once[ifile] = True

        in_fields = {field_name:None for field_name in self.field_names}
        tgt_fields = {field_name:None for field_name in self.field_names}
        
        # WARNING
        # Assume all fields have shape (#time_steps, #particles, # dimension)
        # For internal energy it should be (#time_steps, #particles, 1)
        with h5py.File(self.file_lists[ifile], "r") as f:
            for field_name in self.field_names:
                # Each field has shape (#time_steps, #particles, # dimension)
                in_fields[field_name] = f[field_name][start_idx:end_idx].astype(np.float32)
                tgt_fields[field_name] = f[field_name][end_idx:end_idx+1].astype(np.float32)
                
                # If this is InternalEnergy and it's not already 3D, make it 3D
                if field_name == 'InternalEnergy' and len(in_fields[field_name].shape) == 2:
                    in_fields[field_name] = in_fields[field_name][..., np.newaxis]
                    tgt_fields[field_name] = tgt_fields[field_name][..., np.newaxis]

        in_fields = {key:torch.from_numpy(field).float() for key, field in in_fields.items()}
        tgt_fields = {key:torch.from_numpy(field).float() for key, field in tgt_fields.items()}
        
        # If self.norm is not None:
        if self.augment:
            # Apply time reversal symmetry
            # Field shape (#n_timesteps, #n_particles, #dims)
            if np.random.random() < self.augment_prob:
                flip_dim = 0
                for i, (key, field) in enumerate(in_fields.items()):
                    if key == "Velocities":
                        field = -1 * field
                    in_fields[key] = torch.flip(field, [flip_dim])


            # Apply random permutaion of xyz axis
            if np.random.random() < self.augment_prob:
                perm_idx = torch.randperm(3)
                for i, (key, field) in enumerate(in_fields.items()):
                    ndim = self.ndims[i]
                    # Add dimension check, since permuation of axes only support ndim>=2
                    if ndim >=2 and field.shape[-1] == 3:
                        field = field[..., perm_idx]
                        in_fields[key] = field
                for i, (key, field) in enumerate(tgt_fields.items()):
                    ndim = self.ndims[i]
                    if ndim >=2 and field.shape[-1] == 3:
                        field = field[..., perm_idx]
                        tgt_fields[key] = field

        return {
            "input": {
                **in_fields,
                "box_size": torch.tensor([self.box_size], dtype=torch.float32),
                "dt": torch.tensor([self.dt], dtype=torch.float32)
            },
            "target": tgt_fields,
        }
