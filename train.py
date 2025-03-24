import os
import torch
import json
import numpy as np
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataloader import SequenceDataset 
from data_utils import preprocess
from graph_network import EncodeProcessDecode  

def rollout(model, data, metadata, noise_std):
    device = next(model.parameters()).device
    model.eval()
    window_size = 6  

    total_time = data["Coordinates"].size(0)
    traj = data["Coordinates"][:window_size].permute(1, 0, 2)
    particle_type = None

    for time in range(total_time - window_size):
        # Build a graph with no noise for rollout
        from data_utils import preprocess
        graph = preprocess(particle_type, traj[:, -window_size:], None, metadata, 0.0)
        graph = graph.to(device)

        # Predict acceleration
        acceleration = model(graph).cpu()

        # Un-normalize acceleration
        acc_std = torch.tensor(metadata["acc_std"])
        acc_mean = torch.tensor(metadata["acc_mean"])
        acceleration = acceleration * torch.sqrt(acc_std**2 + noise_std**2) + acc_mean

        recent_position = traj[:, -1]
        recent_velocity = recent_position - traj[:, -2]
        new_velocity = recent_velocity + acceleration
        new_position = recent_position + new_velocity

        # Append new position
        traj = torch.cat((traj, new_position.unsqueeze(1)), dim=1)

    # shape: [num_particles, total_time, dim] -> transpose -> [total_time, num_particles, dim]
    return traj.permute(1, 0, 2)

def train():
    batch_size = 2
    num_epochs = 1
    noise_std = 3e-4
    learning_rate = 1e-4
    model_path = "model_output"
    os.makedirs(model_path, exist_ok=True)
    
    # Load metadata
    with open('/Users/matthewpan/Desktop/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Initialize dataset
    dataset = SequenceDataset(
        paths=["/Users/matthewpan/Desktop/fullrun.hdf5"],
        window_size=5
        # fields = ['Coordinates', 'Velocities', 'HydroAcceleration'] #fix SequenceDataset 
    )

    # Create data loader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Initialize model
    simulator = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=3  
    ).cuda()

    optimizer = torch.optim.Adam(simulator.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        simulator.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        count = 0

        for batch in bar:
            # Process each sample in the batch to create graphs
            graphs = [preprocess(
                particle_type=None,
                position_seq=batch["input"]["Coordinates"][i],
                target_position=batch["target"]["Coordinates"][i],
                metadata=metadata,
                noise_std=noise_std,
                num_neighbors=16
            ) for i in range(len(batch["input"]["Coordinates"]))]
            
            # Stack graphs into a batch
            batch_graph = pyg.data.Batch.from_data_list(graphs).to('cuda')
            
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

    # Save model
    torch.save(simulator.state_dict(), os.path.join(model_path, "model_final.pth"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
