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
        **kwargs,
    ):
        # load simulations, assume the name of hdf5 file is *.hdf5
        #file_lists = [sorted(glob(path)) for path in paths]
        file_lists = paths
        self.file_lists = file_lists
        self.nfiles = len(file_lists)
        
        
        print(f"initializing with {self.nfiles} files")

        if self.nfiles == 0:
            raise FileNotFoundError("file not found for {}".format(paths))
        self.is_read_once = np.full(self.nfiles, False)

        # self.meta_data = []
        # Here we assume all the hdf5 files have same keys and each field inside has the same sequence length and num particles
        with h5py.File(paths[0], "r") as f:
            self.field_names = [field_name for field_name in f.keys()]
            self.num_snapshots = f[self.field_names[0]].shape[0]
            self.num_particles = f[self.field_names[0]].shape[1]
            self.ndims = [f[field_name][:].shape[-1] for field_name in self.field_names]

        self.norms = norms
        self.augment = augment
        self.window_size = window_size
        
        # Assertion 1: Check if the number of snapshots is larger than the window size
        assert self.num_snapshots >= self.window_size + 1, "num_snapshots must be larger than window_size"

        # calculate number of samples for the whole training set, each batch is a sub-sequence with length window_size +1
        self.num_sequences = self.num_snapshots - (self.window_size + 1) + 1
        self.num_samples = self.nfiles * self.num_sequences
        self.start_indices = list(range(self.num_sequences))
        
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

        # check if a file is read once
        if not self.is_read_once[ifile]:
            self.is_read_once[ifile] = True

        in_fields = {field_name:None for field_name in self.field_names}
        tgt_fields = {field_name:None for field_name in self.field_names}
        with h5py.File(self.file_lists[ifile], "r") as f:
            for field_name in self.field_names:
                # each field has shape (#time_steps, #particles, # dimension)
                in_fields[field_name] = f[field_name][start_idx:end_idx]
                tgt_fields[field_name] = f[field_name][end_idx:end_idx+1]

        in_fields = {key:torch.from_numpy(field) for key, field in in_fields.items()}
        tgt_fields = {key:torch.from_numpy(field) for key, field in tgt_fields.items()}

        # if self.norm is not None:
        if self.augment:
            # apply time reversal symmetry
            # field shape (#n_timesteps, #n_particles, #dims)
            if np.random.random() < 0.5:
                flip_dim = 0
                for key, field in in_fields.items():
                    if key == "Velocities":
                        # velocity should be in opposite direction
                        field = -1 * field
                    in_fields[key] = torch.flip(field, [flip_dim])
                # no need for target fields, since we only predict next 1 time step

            # apply random permutaion of xyz axis
            if np.random.random() < 0.5:
                perm_idx = torch.randperm(3)
                for key, field in in_fields.items():
                    field = field[..., perm_idx]
                    in_fields[key] = field
                for key, field in tgt_fields.items():
                    field = field[..., perm_idx]

        return {
            "input": in_fields,
            "target": tgt_fields,
        }
