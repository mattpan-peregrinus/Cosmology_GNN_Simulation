import torch
from dataloader import SequenceDataset
from torch.utils.data import DataLoader

def main():
    dataset = SequenceDataset(
        paths=['/Users/matthewpan/Desktop/fullrun.hdf5'],
        window_size=5,
        augment=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    
    print("Testing data loader ...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print("Input shapes:")
        for key, value in batch['input'].items():
            print(f"{key}: {value.shape}")
        print("\nTarget shapes:")
        for key, value in batch['target'].items():
            print(f"{key}: {value.shape}")
        
        if batch_idx == 0:
            break
        
if __name__ == "__main__":
    main()