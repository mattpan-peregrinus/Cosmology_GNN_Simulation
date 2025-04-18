import os
import torch
import json
import numpy as np
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

torch.set_default_dtype(torch.float32)

from config import get_config
from dataloader import SequenceDataset 
from data_utils import preprocess
from graph_network import EncodeProcessDecode  
from validation_utils import get_train_val_datasets
from validation import validate
from visualization import plot_losses

def rollout(model, data, metadata, noise_std):
    device = next(model.parameters()).device
    model.eval()
    window_size = 6  

    total_time = data["Coordinates"].size(0)
    traj = data["Coordinates"][:window_size].permute(1, 0, 2).float()
    particle_type = None

    for time in range(total_time - window_size):
        # Build a graph with no noise for rollout
        from data_utils import preprocess
        input_positions = traj[:, -window_size:].permute(1, 0, 2)
        graph = preprocess(particle_type, input_positions, None, metadata, 0.0, num_neighbors=16)
        graph = graph.to(device)

        # Predict acceleration
        acceleration = model(graph).cpu()

        # Un-normalize acceleration
        acc_std = torch.tensor(metadata["acc_std"], dtype=torch.float32)
        acc_mean = torch.tensor(metadata["acc_mean"], dtype=torch.float32)
        acceleration = acceleration * torch.sqrt(acc_std**2 + noise_std**2) + acc_mean

        recent_position = traj[:, -1]
        recent_velocity = recent_position - traj[:, -2]
        new_velocity = recent_velocity + acceleration
        new_position = recent_position + new_velocity
        traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

    return traj.permute(1, 0, 2)

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
    
    batch_size = 2
    num_epochs = 1
    noise_std = 3e-4
    learning_rate = 1e-4
    model_path = "model_output"
    os.makedirs(model_path, exist_ok=True)
    
    train_dataset, val_dataset = get_train_val_datasets(
        data_path=args.dataset_path,
        window_size=args.window_size,
        val_split=0.2,  # 20% for validation
        augment=args.augment_prob > 0, 
        augment_prob=args.augment_prob,
        seed=args.seed
    )

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
        'temperature': {'train': [], 'val': []}
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
        temp_loss_total = 0.0
        count = 0

        for batch in bar:
            for key in batch["input"]:
                batch["input"][key] = batch["input"][key].float()
            for key in batch["target"]:
                batch["target"][key] = batch["target"][key].float()
                
            #print(f"Batch input Coordinates shape: {batch['input']['Coordinates'].shape}")
            #print(f"Batch target Coordinates shape: {batch['target']['Coordinates'].shape}")
            
            if "InternalEnergy" not in batch["input"]:
                raise ValueError("InternalEnergy is required in the dataset")
            #print(f"Batch input InternalEnergy shape: {batch['input']['InternalEnergy'].shape}")
            
            # Process each sample in the batch to create graphs
            graphs = []
            for i in range(len(batch["input"]["Coordinates"])):
                input_coords = batch["input"]["Coordinates"][i] 
                target_coords = batch["target"]["Coordinates"][i] 
                temperature_seq = batch["input"]["InternalEnergy"][i]  
                
                graph = preprocess(
                    particle_type=None,
                    position_seq=input_coords,
                    target_position=target_coords,
                    metadata=args.metadata,
                    noise_std=args.noise_std,
                    num_neighbors=args.num_neighbors,
                    temperature_seq=temperature_seq,
                    target_temperature=batch["target"]["InternalEnergy"][i],
                )
                graphs.append(graph)
            
            # Stack graphs into a batch
            batch_graph = pyg.data.Batch.from_data_list(graphs).to(device)
            
            # Forward pass
            predictions = simulator(batch_graph)
            acc_pred = predictions['acceleration']
            temp_pred = predictions['temperature']
            
            if batch_graph.y_acc is not None:
                acc_loss = loss_fn(acc_pred, batch_graph.y_acc)
            else:
                acc_loss = torch.tensor(0.0, device=device)
            
            if hasattr(batch_graph, 'y_temp') and batch_graph.y_temp is not None:
                if temp_pred.shape != batch_graph.y_temp.shape:
                    print(f"Shape mismatch in loss: temp_pred {temp_pred.shape}, y_temp {batch_graph.y_temp.shape}")
                    temp_loss = loss_fn(temp_pred.view(batch_graph.y_temp.shape), batch_graph.y_temp)
                else:
                    temp_loss = loss_fn(temp_pred, batch_graph.y_temp)
            else:
                temp_loss = torch.tensor(0.0, device=device)
            
            acc_loss = loss_fn(acc_pred, batch_graph.y_acc)
            temp_loss = loss_fn(temp_pred, batch_graph.y_temp)
            
            combined_loss = args.acc_loss_weight * acc_loss + args.temp_loss_weight * temp_loss

            # Backward pass
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            acc_loss_total += acc_loss.item()
            temp_loss_total += temp_loss.item()
            count += 1
            
            bar.set_postfix({
            "loss": combined_loss.item(), 
            "avg_loss": total_loss / count,
            "acc_loss": acc_loss.item(),
            "temp_loss": temp_loss.item()
            })
            
            global_step += 1
        
        avg_train_loss = total_loss / count if count > 0 else float('inf')
        avg_acc_train_loss = acc_loss_total / count if count > 0 else float('inf')
        avg_temp_train_loss = temp_loss_total / count if count > 0 else float('inf')
        
        train_losses.append(avg_train_loss)
        component_losses['acceleration']['train'].append(avg_acc_train_loss)
        component_losses['temperature']['train'].append(avg_temp_train_loss)
        
        val_loss, val_component_losses = validate(simulator, val_loader, device, loss_fn, args.acc_loss_weight, args.temp_loss_weight, args.metadata, 0, args.num_neighbors)
        
        val_losses.append(val_loss)
        component_losses['acceleration']['val'].append(val_component_losses['acceleration'])
        component_losses['temperature']['val'].append(val_component_losses['temperature'])
        
        print(f"Epoch {epoch}: "
              f"training loss = {avg_train_loss:.6f}, "
              f"validation loss = {val_loss:.6f}, "
              f"train acc loss = {avg_acc_train_loss:.6f}, "
              f"val acc loss = {val_component_losses['acceleration']:.6f}, "
              f"train temp loss = {avg_temp_train_loss:.6f}, "
              f"val temp loss = {val_component_losses['temperature']:.6f}")
        
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
            'temp_train': component_losses['temperature']['train'],
            'temp_val': component_losses['temperature']['val']
        },
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return simulator
    
    
    
if __name__ == "__main__":
    train()
