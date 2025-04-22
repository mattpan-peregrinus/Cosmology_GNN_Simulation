import os
import argparse
import json
import torch 

def get_config():
    parser = argparse.ArgumentParser(description='Cosmology GNN Simulation')
    
    # Necessary args
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset file')
    parser.add_argument('--metadata_path', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--output_dir', type=str, default='model_output', help='Path to output directory')
    
    parser.add_argument('--num_neighbors', type=int, default=16, help='Number of nearest neighbors to consider for each node')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    
    # Training / hardware args 
    parser.add_argument('--window_size', type=int, default=5, help='Number of time steps to use for input sequence')
    parser.add_argument('--latent_size', type=int, default=128, help='Size of latent representations')
    parser.add_argument('--mlp_hidden_size', type=int, default=128, help='Hidden size for MLPs')
    parser.add_argument('--mlp_num_hidden_layers', type=int, default=2, help='Number of hidden layers in MLPs')
    parser.add_argument('--num_message_passing_steps', type=int, default=10, help='Number of message passing steps in processor')
    parser.add_argument('--output_size', type=int, default=3, help='Output dimension (typically 3 for 3D acceleration)')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Standard deviation of noise added to positions')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--save_every', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--augment_prob', type=float, default=0.0, help='Probability of applying augmentations (0.0-1.0)')
    parser.add_argument('--temp_loss_weight', type=float, default=1.0, help='Weight for temperature loss in combined loss')
    parser.add_argument('--acc_loss_weight', type=float, default=1.0, help='Weight for acceleration loss in combined loss')
    
    
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--plots_dir', type=str, default='plots', help='Subdirectory for saving plots')
    parser.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model for fine-tuning')
    parser.add_argument('--test_data_path', type=str, default=None, help='Path to test data for relative error evaluation')
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.metadata_path, 'r') as f:
        args.metadata = json.load(f)
        
    import numpy as np
    import random 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    torch.set_default_dtype(torch.float32)
    
    return args