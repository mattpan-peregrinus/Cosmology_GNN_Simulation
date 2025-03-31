import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py
from tqdm import tqdm
import argparse
from data_utils import preprocess
from graph_network import EncodeProcessDecode


def predict_rollout(model, initial_positions, metadata, num_rollout_steps, device="cpu"):
    """
    Use the model to predict future positions given initial positions.
    
    Args:
        model: The trained EncodeProcessDecode model
        initial_positions: Initial positions [time_steps, num_particles, 3]
        metadata: Simulation metadata
        num_rollout_steps: Number of steps to predict forward
        device: Device to run the model on
    
    Returns:
        predicted_trajectory: Full trajectory including initial positions
    """
    model.eval()
    
    # Convert to torch if it's numpy
    if isinstance(initial_positions, np.ndarray):
        initial_positions = torch.tensor(initial_positions, dtype=torch.float32)
    
    # Transpose to [num_particles, time_steps, 3] for processing
    initial_positions_t = initial_positions.permute(1, 0, 2)
    
    # Start with initial positions
    trajectory = initial_positions_t.clone()
    
    # Calculate window size (number of input frames)
    window_size = initial_positions.shape[0]
    
    print(f"Predicting rollout for {num_rollout_steps} steps...")
    for step in tqdm(range(num_rollout_steps)):
        # Get the most recent window of positions
        positions = trajectory[:, -window_size:]
        
        # Compute velocities (inputs for the model)
        velocities = positions[:, 1:] - positions[:, :-1]
        
        # For rollout, we don't need target position
        target_position = None
        
        try:
            # Create graph - using a simplified version to match model expectations
            from torch_geometric.nn import knn_graph
            from torch_geometric.data import Data
            
            # Recent position is the last frame
            recent_position = positions[:, -1]
            
            # Create KNN graph
            edge_index = knn_graph(recent_position, k=16, loop=True)
            
            # Create node features - CUSTOMIZE THIS BASED ON YOUR MODEL'S EXPECTATIONS
            # This is a simplified placeholder
            particle_type = torch.zeros(recent_position.shape[0], dtype=torch.float32)
            
            # Reshape velocities - specific to your model's expected format
            velocity_features = velocities.reshape(velocities.shape[0], -1)
            
            # Add dimensions to match what your model expects
            expected_feature_size = 18  # This should match what your model expects
            node_features = torch.zeros((recent_position.shape[0], expected_feature_size), 
                                        dtype=torch.float32)
            
            # Fill what we can with real data
            min_cols = min(velocity_features.shape[1], expected_feature_size)
            node_features[:, :min_cols] = velocity_features[:, :min_cols]
            
            # Create edge features
            senders, receivers = edge_index
            edge_displacement = recent_position[senders] - recent_position[receivers]
            edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)
            edge_attr = torch.cat([edge_displacement, edge_distance], dim=-1)
            
            # Create graph
            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr
            ).to(device)
            
            # Predict acceleration
            with torch.no_grad():
                acceleration = model(graph).cpu()
            
            # Un-normalize acceleration if needed
            if "acc_mean" in metadata and "acc_std" in metadata:
                acc_mean = torch.tensor(metadata["acc_mean"])
                acc_std = torch.tensor(metadata["acc_std"])
                acceleration = acceleration * acc_std + acc_mean
                
            # Update positions
            recent_velocity = positions[:, -1] - positions[:, -2]
            next_velocity = recent_velocity + acceleration
            next_position = positions[:, -1] + next_velocity
            
            # Add new position to trajectory
            trajectory = torch.cat([trajectory, next_position.unsqueeze(1)], dim=1)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            break
    
    # Transpose back to [time_steps, num_particles, 3]
    return trajectory.permute(1, 0, 2)


def create_rollout_animation(ground_truth, prediction, metadata, 
                            output_path="particle_rollout.mp4", step_stride=1, 
                            fps=30, dpi=100):
    """
    Create an animation comparing ground truth and prediction.
    
    Args:
        ground_truth: Ground truth trajectory [time_steps, num_particles, 3]
        prediction: Predicted trajectory [time_steps, num_particles, 3]
        metadata: Simulation metadata
        output_path: Path to save the animation
        step_stride: Steps to skip for faster animation
        fps: Frames per second for the animation
        dpi: DPI for the figure
    """
    # Convert to numpy if needed
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy()
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    
    # Determine number of steps and particles
    num_steps = min(ground_truth.shape[0], prediction.shape[0])
    num_particles = ground_truth.shape[1]
    
    # Set up the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    
    # Get bounds from metadata
    if "bounds" in metadata:
        bounds = np.array(metadata["bounds"])
        # Handle different shapes
        if len(bounds.shape) == 3:
            x_bounds = [bounds[0, 0, 0], bounds[0, 1, 0]]
            y_bounds = [bounds[1, 0, 0], bounds[1, 1, 0]]
        else:
            x_bounds = [bounds[0, 0], bounds[0, 1]]
            y_bounds = [bounds[1, 0], bounds[1, 1]]
    else:
        # Compute bounds from data
        min_pos = np.min(np.concatenate([ground_truth[:, :, :2], prediction[:, :, :2]]), axis=(0, 1))
        max_pos = np.max(np.concatenate([ground_truth[:, :, :2], prediction[:, :, :2]]), axis=(0, 1))
        margin_x = 0.1 * (max_pos[0] - min_pos[0])
        margin_y = 0.1 * (max_pos[1] - min_pos[1])
        x_bounds = [min_pos[0] - margin_x, max_pos[0] + margin_x]
        y_bounds = [min_pos[1] - margin_y, max_pos[1] + margin_y]
    
    # Set up each subplot
    for i, (ax, title, data) in enumerate(zip(
            axes, ["Ground Truth", "Prediction"], [ground_truth, prediction])):
        ax.set_title(title)
        ax.set_xlim(x_bounds[0], x_bounds[1])
        ax.set_ylim(y_bounds[0], y_bounds[1])
        ax.set_aspect('equal')
        
        # Create scatter plot for particles
        scatter = ax.scatter([], [], s=5, c='blue')
        axes[i].scatter = scatter
    
    # Add a frame counter
    frame_text = fig.text(0.02, 0.02, '', fontsize=10)
    
    # Animation update function
    def update(frame_idx):
        # Use stride to skip frames
        step = frame_idx * step_stride
        frame_text.set_text(f'Frame: {step}')
        
        # Update each plot
        for i, data in enumerate([ground_truth, prediction]):
            if step < data.shape[0]:
                axes[i].scatter.set_offsets(data[step, :, :2])
        
        return [axes[0].scatter, axes[1].scatter, frame_text]
    
    # Create the animation
    frames = np.arange(0, num_steps, step_stride)
    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000/fps, blit=True)
    
    # Save the animation
    print(f"Saving animation to {output_path}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Animation saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize particle simulation with model predictions')
    parser.add_argument('--hdf5', type=str, required=True, help='Path to HDF5 file with simulation data')
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata.json')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='model_prediction.mp4', help='Output video path')
    parser.add_argument('--initial_steps', type=int, default=10, help='Number of initial steps for prediction')
    parser.add_argument('--rollout_steps', type=int, default=50, help='Number of steps to predict')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second in output video')
    parser.add_argument('--step_stride', type=int, default=1, help='Stride for animation frames')
    
    args = parser.parse_args()
    
    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=3
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model}")
    
    # Load ground truth trajectory
    with h5py.File(args.hdf5, "r") as f:
        ground_truth = f['Coordinates'][:]
        print(f"Loaded trajectory with shape: {ground_truth.shape}")
    
    # Get initial positions for prediction
    initial_positions = ground_truth[:args.initial_steps]
    full_ground_truth = ground_truth[:args.initial_steps + args.rollout_steps]
    
    # Predict rollout
    try:
        predicted_trajectory = predict_rollout(
            model, 
            initial_positions, 
            metadata, 
            args.rollout_steps,
            device
        )
        
        # Create animation
        create_rollout_animation(
            full_ground_truth,
            predicted_trajectory,
            metadata,
            output_path=args.output,
            step_stride=args.step_stride,
            fps=args.fps
        )
    except Exception as e:
        print(f"Error during visualization: {e}")
        
        # Fall back to just visualizing ground truth
        print("Falling back to visualizing only ground truth...")
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Get bounds
        if "bounds" in metadata:
            bounds = np.array(metadata["bounds"])
            if len(bounds.shape) == 3:
                ax.set_xlim(bounds[0, 0, 0], bounds[0, 1, 0])
                ax.set_ylim(bounds[1, 0, 0], bounds[1, 1, 0])
            else:
                ax.set_xlim(bounds[0, 0], bounds[0, 1])
                ax.set_ylim(bounds[1, 0], bounds[1, 1])
        
        ax.set_title("Ground Truth Trajectory")
        scatter = ax.scatter([], [], s=5, c='blue')
        
        def update(frame):
            scatter.set_offsets(ground_truth[frame, :, :2])
            return [scatter]
        
        anim = animation.FuncAnimation(
            fig, update, frames=len(ground_truth), interval=1000/args.fps, blit=True)
        
        anim.save("ground_truth_only.mp4", writer=animation.FFMpegWriter(fps=args.fps))
        plt.close(fig)
        print("Ground truth animation saved to ground_truth_only.mp4")


if __name__ == "__main__":
    main()