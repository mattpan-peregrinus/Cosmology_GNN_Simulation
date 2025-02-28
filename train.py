import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from dataset import OneStepDataset, RolloutDataset
from model import EncodeProcessDecode  # <-- Use the EncodeProcessDecode class

def rollout(model, data, metadata, noise_std):
    device = next(model.parameters()).device
    model.eval()
    # We'll define a local window size (e.g., 5 frames + 1 target => 6).
    window_size = 6  

    total_time = data["position"].size(0)
    # shape: [window_size, num_particles, dim] -> transpose -> [num_particles, window_size, dim]
    traj = data["position"][:window_size].permute(1, 0, 2)
    particle_type = data["particle_type"]

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--noise', type=float, default=3e-4)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    os.makedirs(args.model_path, exist_ok=True)

    # Load dataset
    from dataset import OneStepDataset, RolloutDataset
    train_dataset = OneStepDataset(args.data_path, "train", noise_std=args.noise)
    valid_dataset = OneStepDataset(args.data_path, "valid", noise_std=args.noise)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    valid_rollout_dataset = RolloutDataset(args.data_path, "valid")

    metadata = train_dataset.metadata

    # Build the EncodeProcessDecode model with GraphSAGE inside
    simulator = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=2  # e.g. 2D (x,y) acceleration or position delta
    ).cuda()

    optimizer = torch.optim.Adam(simulator.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    # Training loop
    global_step = 0
    for epoch in range(args.epoch):
        simulator.train()
        bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        total_loss = 0.0
        count = 0

        for batch in bar:
            batch = batch.to('cuda')
            pred = simulator(batch)  # shape: [num_nodes, 2]
            loss = loss_fn(pred, batch.y)  # batch.y should match shape [num_nodes, 2]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1
            bar.set_postfix({"loss": loss.item(), "avg_loss": total_loss / count})
            global_step += 1

        # Evaluate on the valid set (one-step)
        with torch.no_grad():
            simulator.eval()
            val_loss = 0.0
            val_count = 0
            for vbatch in valid_loader:
                vbatch = vbatch.to('cuda')
                vpred = simulator(vbatch)
                vloss = loss_fn(vpred, vbatch.y)
                val_loss += vloss.item()
                val_count += 1
            val_loss /= max(val_count, 1)
            print(f"Epoch {epoch}: validation loss = {val_loss}")

    # Save model
    torch.save(simulator.state_dict(), os.path.join(args.model_path, "model_final.pth"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
