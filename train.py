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
        graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0)
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

def train():
    args = get_config()
    device = torch.device(args.device)
    
    batch_size = 2
    num_epochs = 1
    noise_std = 3e-4
    learning_rate = 1e-4
    model_path = "model_output"
    os.makedirs(model_path, exist_ok=True)
    
    dataset = SequenceDataset(
        paths=[args.dataset_path],
        window_size=args.window_size,
        augment=args.augment
    )

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
            
    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    global_step = 0
    for epoch in range(args.num_epochs):
        simulator.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        count = 0

        for batch in bar:
            for key in batch["input"]:
                batch["input"][key] = batch["input"][key].float()
            for key in batch["target"]:
                batch["target"][key] = batch["target"][key].float()
            # Process each sample in the batch to create graphs
            graphs = [preprocess(
                particle_type=None,
                position_seq=batch["input"]["Coordinates"][i],
                target_position=batch["target"]["Coordinates"][i][0],
                metadata=args.metadata,
                noise_std=args.noise_std,
                num_neighbors=args.num_neighbors
            ) for i in range(len(batch["input"]["Coordinates"]))]
            
            # Stack graphs into a batch
            batch_graph = pyg.data.Batch.from_data_list(graphs).to(device)
            
            # Forward pass
            pred = simulator(batch_graph)
            loss = loss_fn(pred, batch_graph.y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / count})
            global_step += 1
            
        print(f"Epoch {epoch}: training loss = {total_loss / count}")
        
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch}.pth")
            torch.save(simulator.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")
        
    final_model_path = os.path.join(args.output_dir, "model_final.pth")
    torch.save(simulator.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")
    
if __name__ == "__main__":
    train()
