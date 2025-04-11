import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses(train_losses, val_losses, save_path, component_losses=None):
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot main losses
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # If component losses are provided, plot them separately
    if component_losses is not None:
        plt.figure(figsize=(12, 8))
        
        for component, losses in component_losses.items():
            if 'train' in losses:
                plt.plot(epochs, losses['train'], linewidth=2, 
                         label=f'Training {component} Loss')
            if 'val' in losses:
                plt.plot(epochs, losses['val'], linewidth=2, linestyle='--',
                         label=f'Validation {component} Loss')
        
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Component Loss', fontsize=14)
        plt.title('Component-wise Losses', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Save component loss plot
        component_path = save_path.replace('.png', '_components.png')
        plt.tight_layout()
        plt.savefig(component_path)
        plt.close()