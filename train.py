import os
import torch
import json
import numpy as np
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR

torch.set_default_dtype(torch.float32)

from config import get_config
from dataloader import SequenceDataset 
from data_utils import preprocess
from graph_network import EncodeProcessDecode  
from validation import validate
import h5py
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, output_path, component_losses, learning_rates):
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    gs = plt.GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
        
    # Plot 1: Combined training and validation loss (top span)
    ax1 = fig.add_subplot(gs[0, :])
        
    # Plot 2-3: Component losses (middle row)
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
        
    # Plot 4: Learning rate (bottom span)
    ax4 = fig.add_subplot(gs[2, :])
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Training and Validation Loss per Epoch', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    ax1.set_yscale('log')
    
    epochs = range(1, len(train_losses) + 1)
    
    # Main loss plot
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Training and Validation Loss per Epoch', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    ax1.set_yscale('log')
    
    # Acceleration loss
    ax2.plot(epochs, component_losses['acceleration']['train'], 'b-', linewidth=2, label='Train')
    ax2.plot(epochs, component_losses['acceleration']['val'], 'r-', linewidth=2, label='Validation')
    ax2.set_title('Acceleration Loss', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss (MSE)', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.set_yscale('log')
        
    ax3.plot(epochs, component_losses['temp_rate']['train'], 'b-', linewidth=2, label='Train')
    ax3.plot(epochs, component_losses['temp_rate']['val'], 'r-', linewidth=2, label='Validation')
    ax3.set_title('Temperature_Rate Loss', fontsize=14)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Loss (MSE)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(fontsize=10)
    ax3.set_yscale('log')
        
    ax4.plot(epochs, learning_rates, 'g-', linewidth=3, label='Learning Rate')
    ax4.set_title('Learning Rate Schedule (Exponential Decay)', fontsize=14)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Learning Rate', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_yscale('log')
    ax4.legend(fontsize=10)
            
    # Add text annotations for initial and final LR
    if len(learning_rates) > 0:
        ax4.text(0.02, 0.95, f'Initial LR: {learning_rates[0]:.2e}', 
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    if len(learning_rates) > 1:
        ax4.text(0.02, 0.05, f'Current LR: {learning_rates[-1]:.2e}',
                 transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def load_pretrained_model(model, model_path):
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded pretrained model from {model_path}")
    except Exception as e:
        print(f"Error loading pretrained model: {e}")
    return model

def train():
    args = get_config()
    device = torch.device(args.device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    train_dataset = SequenceDataset(
        paths=args.train_dir,
        window_size=args.window_size,
        metadata=args.metadata,
        augment=args.augment_prob > 0,
        augment_prob=args.augment_prob
    )
    val_dataset = SequenceDataset(
        paths=args.val_dir,
        window_size=args.window_size,
        metadata=args.metadata,
        augment=False
    )
    
    # Obtain dt and box_size from the metadata
    dt = args.metadata["dt"]
    box_size = args.metadata["box_size"]
    print(f"Using time step (dt): {dt}")
    print(f"Using box size: {box_size}")

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Initialize model
    simulator = EncodeProcessDecode(
        latent_size=args.latent_size,
        mlp_hidden_size=args.mlp_hidden_size,
        mlp_num_hidden_layers=args.mlp_num_hidden_layers,
        num_message_passing_steps=args.num_message_passing_steps,
        output_size=args.output_size
    ).to(device)
    
    for param in simulator.parameters():
        if param.data.dtype != torch.float32:
            param.data = param.data.float()
    
    # Load pretrained model if provided
    if args.pretrained_model:
        simulator = load_pretrained_model(simulator, args.pretrained_model)
        print(f"Starting training from pretrained model: {args.pretrained_model}")
    
    # Set up learning rate scheduler 
    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    initial_lr = args.learning_rate
    final_lr = args.final_learning_rate
    decay_rate = (final_lr / initial_lr) ** (1 / args.num_epochs)
    scheduler = ExponentialLR(optimizer, gamma=decay_rate)
    train_learning_rates = []
    
    print(f"Learning rate will decay from {initial_lr} to {final_lr} over {args.num_epochs} epochs.")
    print(f"Decay rate: {decay_rate}")
    
    loss_fn = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []
    component_losses = {
        'acceleration': {'train': [], 'val': []},
        'temp_rate': {'train': [], 'val': []}
    }
    
    best_val_loss = float('inf')
    best_epoch = -1
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        train_learning_rates.append(current_lr)
        simulator.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        acc_loss_total = 0.0
        temp_rate_loss_total = 0.0
        count = 0

        for batch in bar:
            for key in batch["input"]:
                batch["input"][key] = batch["input"][key].float()
            for key in batch["target"]:
                batch["target"][key] = batch["target"][key].float()
            
            # Process each sample in the batch to create graphs
            graphs = []
            for i in range(len(batch["input"]["Coordinates"])):
                input_coords = batch["input"]["Coordinates"][i] 
                target_coords = batch["target"]["Coordinates"][i] 
                input_temperature = batch["input"]["InternalEnergy"][i]
                target_temperature = batch["target"]["InternalEnergy"][i]  
                
                graph = preprocess(
                    position_seq=input_coords,
                    target_position=target_coords,
                    temperature_seq=input_temperature,
                    target_temperature=target_temperature,
                    metadata=args.metadata,
                    noise_std=args.noise_std,
                    num_neighbors=args.num_neighbors,
                    dt=dt,
                    box_size=box_size
                )
                graphs.append(graph)
            
            # Stack graphs into a batch
            batch_graph = pyg.data.Batch.from_data_list(graphs).to(device)
            
            # Forward pass
            predictions = simulator(batch_graph)
            acc_pred = predictions['acceleration']
            temp_rate_pred = predictions['temp_rate']
            
            acc_loss = loss_fn(acc_pred, batch_graph.y_acc)
            temp_rate_loss = loss_fn(temp_rate_pred, batch_graph.y_temp_rate)
            combined_loss = args.acc_loss_weight * acc_loss + args.temp_rate_loss_weight * temp_rate_loss

            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            acc_loss_total += acc_loss.item()
            temp_rate_loss_total += temp_rate_loss.item()
            count += 1
            
            # Only update the progress bar every 10 batches
            if count % 10 == 0 or count == len(train_loader):
                bar.set_postfix({
                    "loss": combined_loss.item(), 
                    "avg_loss": total_loss / count,
                    "acc_loss": acc_loss.item(),
                    "temp_rate_loss": temp_rate_loss.item()
                })
            
            global_step += 1
        
        avg_train_loss = total_loss / count if count > 0 else float('inf')
        avg_acc_train_loss = acc_loss_total / count if count > 0 else float('inf')
        avg_temp_rate_train_loss = temp_rate_loss_total / count if count > 0 else float('inf')
        
        train_losses.append(avg_train_loss)
        component_losses['acceleration']['train'].append(avg_acc_train_loss)
        component_losses['temp_rate']['train'].append(avg_temp_rate_train_loss)
        
        val_loss, val_component_losses = validate(
            simulator, 
            val_loader, 
            device, 
            loss_fn, 
            args.acc_loss_weight, 
            args.temp_rate_loss_weight, 
            args.metadata, 
            0,  #setting noise sd to zero, for now
            args.num_neighbors,
            dt = dt,
            box_size = box_size
        )
        
        val_losses.append(val_loss)
        component_losses['acceleration']['val'].append(val_component_losses['acceleration'])
        component_losses['temp_rate']['val'].append(val_component_losses['temp_rate'])
        
        # Update learning rate 
        scheduler.step()
        
        print(f"Epoch {epoch}: "
              f"training loss = {avg_train_loss:.6f}, "
              f"validation loss = {val_loss:.6f}, "
              f"learning rate = {current_lr:.2e}, "
              f"train acc loss = {avg_acc_train_loss:.6f}, "
              f"val acc loss = {val_component_losses['acceleration']:.6f}, "
              f"train temp_rate loss = {avg_temp_rate_train_loss:.6f}, "
              f"val temp_rate loss = {val_component_losses['temp_rate']:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model
            best_model_path = os.path.join(args.output_dir, "model_best.pth")
            torch.save(simulator.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")

        plot_losses(
            train_losses, 
            val_losses, 
            os.path.join(plots_dir, f'losses_epoch_{epoch}.png'),
            component_losses,
            train_learning_rates
        )
       
        # Periodic checkpoints        
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pth")
            torch.save(simulator.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
    plot_losses(
        train_losses, 
        val_losses, 
        os.path.join(plots_dir, 'losses_final.png'),
        component_losses,
        train_learning_rates
    )
    
    best_model_path = os.path.join(args.output_dir, "model_best.pth")
    if os.path.exists(best_model_path):
        simulator.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from epoch {best_epoch} with validation loss {best_val_loss:.6f}")
        
    final_model_path = os.path.join(args.output_dir, "model_final.pth")
    torch.save(simulator.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'learning_rates': train_learning_rates,
        'component_losses': {
            'acc_train': component_losses['acceleration']['train'],
            'acc_val': component_losses['acceleration']['val'],
            'temp_rate_train': component_losses['temp_rate']['train'],
            'temp_rate_val': component_losses['temp_rate']['val']
        },
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return simulator
    
if __name__ == "__main__":
    train()
