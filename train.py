import os
import torch
import json
import numpy as np
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float32)

from config import get_config
from dataloader import SequenceDataset 
from data_utils import preprocess
from graph_network import EncodeProcessDecode  
from validation_utils import get_train_val_datasets
from validation import validate
import h5py
import matplotlib.pyplot as plt

def plot_losses(train_losses, val_losses, output_path, component_losses=None):
    # Create figure with subplots
    if component_losses:
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Plot 1: Combined training and validation loss (top span)
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot 2-3: Component losses (bottom row)
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
    else:
        # Just create a single plot
        fig, ax1 = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Training and Validation Loss per Epoch', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Loss', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=12)
    
    # Plot component losses if provided
    if component_losses:
        # Acceleration loss
        ax2.plot(epochs, component_losses['acceleration']['train'], 'b-', linewidth=2, label='Train')
        ax2.plot(epochs, component_losses['acceleration']['val'], 'r-', linewidth=2, label='Validation')
        ax2.set_title('Acceleration Loss', fontsize=14)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (MSE)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(fontsize=10)
        
        # Temperature_rate loss
        ax3.plot(epochs, component_losses['temp_rate']['train'], 'b-', linewidth=2, label='Train')
        ax3.plot(epochs, component_losses['temp_rate']['val'], 'r-', linewidth=2, label='Validation')
        ax3.set_title('Temperature_Rate Loss', fontsize=14)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Loss (MSE)', fontsize=12)
        ax3.grid(True, linestyle='--', alpha=0.7)
        ax3.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
def plot_rollout_relative_error(model, test_data_path, metadata, output_path, window_size=5, num_steps=100, device='cpu', clip_value=1000):
    
    with h5py.File(test_data_path, 'r') as f:
        dt = args.metadata.get["dt"]
        box_size = args.metadata.get["box_size"]
        print(f"Using dataset parameters: dt={dt}, box_size={box_size}")
        
        ground_truth = {
            "Coordinates": torch.tensor(f["Coordinates"][:], dtype=torch.float32),
            "InternalEnergy": torch.tensor(f["InternalEnergy"][:], dtype=torch.float32)
        }
        # Is this necessary? 
        if len(ground_truth["InternalEnergy"].shape) == 2:
            ground_truth["InternalEnergy"] = ground_truth["InternalEnergy"].unsqueeze(-1)
        
    # Perform rollout
    print("Performing rollout for relative error calculation...")
    rollout_data = rollout(
        model=model,
        data=ground_truth,
        metadata=metadata,
        noise_std=0.0,  # set noise std to 0 for rollout
        dt=dt,
        box_size=box_size
    )
    
    # Calculate relative errors over time
    pred_coords = rollout_data["Coordinates"][window_size:]
    pred_temps = rollout_data["InternalEnergy"][window_size:]
    
    max_steps = min(len(pred_coords), len(ground_truth["Coordinates"]) - window_size)
    true_coords = ground_truth["Coordinates"][window_size:window_size+max_steps]
    true_temps = ground_truth["InternalEnergy"][window_size:window_size+max_steps]
    
    # Calculate relative error at each timestep
    epsilon = 1e-8 
    
    coord_rel_errors = []
    temp_rel_errors = []
    
    print("Calculating relative errors...")
    for t in range(max_steps):
        # For coordinates (position)
        rel_err = (pred_coords[t] - true_coords[t]) / (torch.abs(true_coords[t]) + epsilon)
        # Calculate absolute relative error and clip extreme values
        rel_err_mean = torch.mean(torch.abs(rel_err)).item() * 100  
        rel_err_mean = min(rel_err_mean, clip_value)  
        coord_rel_errors.append(rel_err_mean)
        
        # For temperature
        true_temp = true_temps[t]
        # check if this is necessary 
        if len(true_temp.shape) == 1:
            true_temp = true_temp.unsqueeze(-1)
            
        pred_temp = pred_temps[t]
        # this too
        if len(pred_temp.shape) == 1:
            pred_temp = pred_temp.unsqueeze(-1)
            
        temp_rel_err = (pred_temp - true_temp) / (torch.abs(true_temp) + epsilon)
        temp_rel_err_mean = torch.mean(torch.abs(temp_rel_err)).item() * 100
        temp_rel_err_mean = min(temp_rel_err_mean, clip_value)  
        temp_rel_errors.append(temp_rel_err_mean)
    
    # Plot relative errors over time
    plt.figure(figsize=(12, 8))
    
    plt.plot(coord_rel_errors, 'b-', linewidth=2, label='Position')
    plt.plot(temp_rel_errors, 'r-', linewidth=2, label='Temperature')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Relative Error Over Time', fontsize=16)
    plt.xlabel('Rollout Step', fontsize=14)
    plt.ylabel('Absolute Relative Error (%) |Ypred - Yreal| / |Yreal|', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Relative error plot saved to {output_path}")

def rollout(model, data, metadata, noise_std, dt, box_size):
    device = next(model.parameters()).device
    model.eval()
    window_size = 6  

    total_time = data["Coordinates"].size(0)
    position_traj = data["Coordinates"][:window_size].permute(1, 0, 2).float() # -> [num_particles, window_size, 3]
    temp_traj = data["InternalEnergy"][:window_size].permute(1, 0, 2).float() # -> [num_particles, window_size, 1]
    
    # Not necessary? If temp_traj is [num_particles, window_size], add a dimension.
    if len(temp_traj.shape) == 2:
        temp_traj = temp_traj.unsqueeze(-1) 

    for time in range(total_time - window_size):
        # Build a graph with no noise for rollout
        input_positions = position_traj[:, -window_size:].permute(1, 0, 2) # -> [window_size, num_particles, 3]
        input_temps = temp_traj[:, -window_size:].permute(1, 0, 2) # -> [window_size, num_particles, 1]
        
        graph = preprocess(
            position_seq=input_positions, 
            temperature_seq=input_temps,
            metadata=metadata, 
            noise_std=0.0, 
            num_neighbors=16, 
            box_size=box_size,
            dt=dt
        )
        graph = graph.to(device)

        # Predict acceleration and temperature rate
        predictions = model(graph)
        acc_pred = predictions['acceleration'].cpu()
        temp_rate_pred = predictions['temp_rate'].cpu()

        # Un-normalize acceleration
        acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
        acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
        
        acc_pred = acc_pred * torch.sqrt(acc_std**2 + noise_std**2) + acc_mean
        
        # Un-normalize temperature rate
        temp_rate_std = torch.tensor(metadata["temp_rate_std"], dtype=torch.float32)
        temp_rate_mean = torch.tensor(metadata["temp_rate_mean"], dtype=torch.float32)
        temp_rate_pred = temp_rate_pred * torch.sqrt(temp_rate_std**2 + noise_std**2) + temp_rate_mean
        
        # PHYSICS INTEGRATION !!!
        # Obtain the recent values of position, velocity, temperature
        recent_position = position_traj[:, -1]
        recent_velocity = (recent_position - position_traj[:, -2]) / dt
        recent_temp = temp_traj[:, -1]
        
        # Integrate to new values, keeping in mind periodicity for position update
        new_velocity = recent_velocity + acc_pred * dt
        new_position = recent_position + new_velocity * dt
        # Perform modulo by box_size to keep particles within the box
        new_position = torch.remainder(new_position, box_size)
        new_temp = recent_temp + temp_rate_pred * dt
        
        position_traj = torch.cat((position_traj, new_position.unsqueeze(1)), dim=1) # -> [num_particles, time_steps + 1, 3]
        temp_traj = torch.cat((temp_traj, new_temp.unsqueeze(1)), dim=1) # -> [num_particles, time_steps + 1, 1]
        
    return {
        "Coordinates": position_traj.permute(1, 0, 2), # -> [total_time_steps, num_particles, 3]
        "InternalEnergy": temp_traj.permute(1, 0, 2) # -> [total_time_steps, num_particles, 1]
    }   

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
    
    train_dataset, val_dataset = get_train_val_datasets(
        data_path=args.dataset_path,
        window_size=args.window_size,
        metadata=args.metadata,
        val_split=0.2,  
        augment=args.augment_prob > 0, 
        augment_prob=args.augment_prob,
        seed=args.seed
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
            
    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        
        print(f"Epoch {epoch}: "
              f"training loss = {avg_train_loss:.6f}, "
              f"validation loss = {val_loss:.6f}, "
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
            component_losses
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
        component_losses
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
        
    if hasattr(args, 'test_data_path') and args.test_data_path:
        print("\nGenerating relative error plot...")
        plot_rollout_relative_error(
            model=simulator,
            test_data_path=args.test_data_path,
            metadata=args.metadata,
            output_path=os.path.join(plots_dir, 'relative_error.png'),
            window_size=args.window_size,
            num_steps=100,  
            device=device,
            clip_value=1000,
        )
    else:
        print("\nNo test data path provided. Skipping relative error plot.")
    
    return simulator
    
if __name__ == "__main__":
    train()
