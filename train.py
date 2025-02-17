import argparse
import collections
import json
import logging
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our converted modules.
import learned_simulator
import noise_utils
import reading_utils  
import connectivity_utils  

# Global constants.
INPUT_SEQUENCE_LENGTH = 6  # The number of timesteps used for inputs (target is the final step).
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3

Stats = collections.namedtuple('Stats', ['mean', 'std'])


def get_kinematic_mask(particle_types):
    """
    Returns a boolean mask that is True for kinematic (obstacle) particles.
    Args:
      particle_types: Tensor of shape [num_particles].
    """
    return (particle_types == KINEMATIC_PARTICLE_ID)


def prepare_inputs(tensor_dict):
    """
    Prepares a single sample for one‐step training.
    Assumes the input dict has a key 'position' with shape 
      [sequence_length, num_particles, dim],
    and transposes it to [num_particles, sequence_length, dim].  
    The target is taken as the final time step.
    Additionally, it adds an extra tensor (with one element) giving the number 
    of particles per example.
    """
    # If there is a batch dimension, remove it.
    if tensor_dict['position'].dim() == 4:
        # Assuming batch size is 1.
        for key in tensor_dict:
            if isinstance(tensor_dict[key], torch.Tensor) and tensor_dict[key].size(0) == 1:
                tensor_dict[key] = tensor_dict[key].squeeze(0)
    
    # Now tensor_dict['position'] should have shape [sequence_length, num_particles, dim]
    pos = tensor_dict['position'].transpose(0, 1)  # now [num_particles, sequence_length, dim]
    target_position = pos[:, -1]  # [num_particles, dim]
    tensor_dict['position'] = pos[:, :-1]  # Remove target time step.
    num_particles = pos.size(0)
    tensor_dict['n_particles_per_example'] = torch.tensor([num_particles])
    if 'step_context' in tensor_dict:
        # Take the penultimate step as context.
        step_context = tensor_dict['step_context'][-2]
        tensor_dict['step_context'] = step_context.unsqueeze(0)
    return tensor_dict, target_position


def prepare_rollout_inputs(context, features):
    """
    Prepares a sample for rollout evaluation.
    Expects a features dict with key 'position' of shape 
      [sequence_length, num_particles, dim].
    Returns a dictionary with transposed positions (inputs), the target positions,
    and a flag that marks this as a trajectory.
    """
    out_dict = dict(context)  # Copy global context if provided.
    pos = features['position'].transpose(0, 1)  # [num_particles, sequence_length, dim]
    target_position = pos[:, -1]
    out_dict['position'] = pos[:, :-1]
    out_dict['n_particles_per_example'] = torch.tensor([pos.size(0)])
    if 'step_context' in features:
        out_dict['step_context'] = features['step_context']
    out_dict['is_trajectory'] = torch.tensor([True], dtype=torch.bool)
    return out_dict, target_position


def get_dataset(data_path, mode, split, metadata):
    """
    Returns a PyTorch Dataset corresponding to the data.
    (Here we assume that reading_utils.get_dataset is implemented to read your TFRecord‐converted data.)
    Args:
      data_path: Path to the dataset directory.
      mode: One of 'one_step_train', 'one_step', or 'rollout'.
      split: One of 'train', 'valid', or 'test'.
    """
    window_length = INPUT_SEQUENCE_LENGTH + 1;
    return reading_utils.get_dataset(
        data_path, mode=mode, split=split, window_length=window_length, metadata=metadata)


def rollout(simulator, features, num_steps, device):
    """
    Rolls out a trajectory by applying the model step‐by‐step.
    Args:
      simulator: Your learned simulator module.
      features: A dict (from the dataset) that must include:
          - 'position': Tensor of shape [num_particles, total_sequence_length, dim],
          - 'n_particles_per_example', and
          - 'particle_type': Tensor of shape [num_particles].
      num_steps: How many rollout steps to run.
      device: The torch.device.
    Returns:
      A dictionary containing:
         - 'initial_positions': the starting positions,
         - 'predicted_rollout': the predicted positions (over time),
         - 'ground_truth_rollout': the true positions,
         - 'particle_types', and, if available,
         - 'global_context'.
    """
    # Use the first INPUT_SEQUENCE_LENGTH steps as the initial input.
    initial_positions = features['position'][:, :INPUT_SEQUENCE_LENGTH]  # [num_particles, INPUT_SEQUENCE_LENGTH, dim]
    ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]  # [num_particles, num_steps, dim]
    global_context = features.get('step_context', None)
    predictions = []
    current_positions = initial_positions.clone().to(device)
    for step in range(num_steps):
        if global_context is None:
            global_context_step = None
        else:
            # Assume global_context is a tensor; select the context for this step and add a batch dimension.
            global_context_step = global_context[step + INPUT_SEQUENCE_LENGTH - 1].unsqueeze(0).to(device)
        with torch.no_grad():
            next_position = simulator(
                current_positions,
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_type'].to(device),
                global_context=global_context_step
            )
        # For kinematic particles, override prediction with ground truth.
        kinematic_mask = get_kinematic_mask(features['particle_type']).to(device)
        next_position_ground_truth = ground_truth_positions[:, step].to(device)
        next_position = torch.where(kinematic_mask.unsqueeze(1), next_position_ground_truth, next_position)
        predictions.append(next_position.cpu())
        # Shift the input sequence: drop the oldest timestep and append the new prediction.
        current_positions = torch.cat([current_positions[:, 1:], next_position.unsqueeze(1)], dim=1)
    output_dict = {
        'initial_positions': initial_positions.transpose(0, 1).cpu(),
        'predicted_rollout': torch.stack(predictions).cpu(),
        'ground_truth_rollout': ground_truth_positions.transpose(0, 1).cpu(),
        'particle_types': features['particle_type'],
    }
    if global_context is not None:
        output_dict['global_context'] = global_context
    return output_dict


def _combine_std(std_x, std_y):
    return np.sqrt(std_x ** 2 + std_y ** 2)


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.load(fp)


def _get_simulator(model_kwargs, metadata, acc_noise_std, vel_noise_std, device):
    """
    Instantiates the simulator.
    Casts the acceleration and velocity statistics from metadata and combines
    them with the noise standard deviation.
    """
    cast = lambda v: np.array(v, dtype=np.float32)
    acceleration_stats = Stats(
        cast(metadata['acc_mean']),
        _combine_std(cast(metadata['acc_std']), acc_noise_std)
    )
    velocity_stats = Stats(
        cast(metadata['vel_mean']),
        _combine_std(cast(metadata['vel_std']), vel_noise_std)
    )
    normalization_stats = {'acceleration': acceleration_stats, 'velocity': velocity_stats}
    if 'context_mean' in metadata:
        context_stats = Stats(cast(metadata['context_mean']), cast(metadata['context_std']))
        normalization_stats['context'] = context_stats

    simulator = learned_simulator.LearnedSimulator(
        num_dimensions=metadata['dim'],
        connectivity_radius=metadata['default_connectivity_radius'],
        graph_network_kwargs=model_kwargs,
        boundaries=metadata['bounds'],
        num_particle_types=NUM_PARTICLE_TYPES,
        normalization_stats=normalization_stats,
        particle_type_embedding_size=16
    ).to(device)
    return simulator


def train_one_step(simulator, dataloader, optimizer, noise_std, device, num_steps, log_interval=100):
    """
    Trains the simulator for a given number of steps.
    For each batch, samples random-walk noise, computes the predicted and target
    normalized accelerations, applies a mask to ignore kinematic particles, computes
    an MSE loss, and runs an optimizer step.`
    """
    simulator.train()
    global_step = 0
    while global_step < num_steps:
        for batch in dataloader:
            # Prepare the inputs (transpose positions, extract target, add n_particles).
            batch, target_position = prepare_inputs(batch)
            print("Batch position shape:", batch['position'].shape) # Debugging print
            # Move all tensor values to device.
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            target_position = target_position.to(device)
            # Sample noise (assumes your noise_utils returns a tensor of same shape as positions).
            sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(batch['position'], noise_std)
            non_kinematic_mask = ~(get_kinematic_mask(batch['particle_type']).to(device))
            noise_mask = non_kinematic_mask.unsqueeze(1).unsqueeze(2).to(batch['position'].dtype)
            sampled_noise = sampled_noise * noise_mask
            # Get predicted and target normalized accelerations.
            pred_acc, target_acc = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_position,
                position_sequence=batch['position'],
                position_sequence_noise=sampled_noise,
                n_particles_per_example=batch['n_particles_per_example'],
                particle_types=batch['particle_type'],
                global_context=batch.get('step_context', None)
            )
            # Compute squared error loss, masking out kinematic particles.
            loss_tensor = (pred_acc - target_acc) ** 2
            mask = (~get_kinematic_mask(batch['particle_type'])).to(loss_tensor.dtype).unsqueeze(1)
            loss_tensor = loss_tensor * mask
            num_non_kinematic = mask.sum()
            loss = loss_tensor.sum() / (num_non_kinematic + 1e-8)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            if global_step % log_interval == 0:
                with torch.no_grad():
                    predicted_next_position = simulator(
                        position_sequence=batch['position'],
                        n_particles_per_example=batch['n_particles_per_example'],
                        particle_types=batch['particle_type'],
                        global_context=batch.get('step_context', None)
                    )
                    pos_mse = ((predicted_next_position - target_position) ** 2).mean().item()
                logging.info(f"Step {global_step}, Loss: {loss.item():.6f}, One-step position MSE: {pos_mse:.6f}")
            if global_step >= num_steps:
                break
    return global_step


def evaluate_one_step(simulator, dataloader, noise_std, device):
    """
    Runs one-step evaluation over the provided data.
    Computes and logs the average loss (masked MSE) and one-step position MSE.
    """
    simulator.eval()
    total_loss = 0.0
    total_samples = 0
    total_pos_mse = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch, target_position = prepare_inputs(batch)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            target_position = target_position.to(device)
            sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(batch['position'], noise_std)
            non_kinematic_mask = ~(get_kinematic_mask(batch['particle_type']).to(device))
            noise_mask = non_kinematic_mask.unsqueeze(1).unsqueeze(2).to(batch['position'].dtype)
            sampled_noise = sampled_noise * noise_mask
            pred_acc, target_acc = simulator.get_predicted_and_target_normalized_accelerations(
                next_position=target_position,
                position_sequence=batch['position'],
                position_sequence_noise=sampled_noise,
                n_particles_per_example=batch['n_particles_per_example'],
                particle_types=batch['particle_type'],
                global_context=batch.get('step_context', None)
            )
            loss_tensor = (pred_acc - target_acc) ** 2
            mask = (~get_kinematic_mask(batch['particle_type'])).to(loss_tensor.dtype).unsqueeze(1)
            loss_tensor = loss_tensor * mask
            num_non_kinematic = mask.sum()
            loss = loss_tensor.sum() / (num_non_kinematic + 1e-8)
            total_loss += loss.item() * num_non_kinematic.item()
            total_samples += num_non_kinematic.item()
            predicted_next_position = simulator(
                position_sequence=batch['position'],
                n_particles_per_example=batch['n_particles_per_example'],
                particle_types=batch['particle_type'],
                global_context=batch.get('step_context', None)
            )
            pos_mse = ((predicted_next_position - target_position) ** 2).sum().item()
            total_pos_mse += pos_mse
    avg_loss = total_loss / (total_samples + 1e-8)
    avg_pos_mse = total_pos_mse / total_samples
    logging.info(f"Evaluation: Loss: {avg_loss:.6f}, One-step position MSE: {avg_pos_mse:.6f}")
    return avg_loss, avg_pos_mse


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the Learned Simulator")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'eval_rollout'],
                        help='Train, one-step evaluation, or rollout evaluation.')
    parser.add_argument('--eval_split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Data split to use for evaluation.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size.')
    parser.add_argument('--num_steps', type=int, default=int(2e7), help='Number of training steps.')
    parser.add_argument('--noise_std', type=float, default=6.7e-4, help='Standard deviation of noise.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path for saving/loading model checkpoints.')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path for saving rollout outputs (required for eval_rollout mode).')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metadata = _read_metadata(args.data_path)
    model_kwargs = {
        'latent_size': 128,
        'mlp_hidden_size': 128,
        'mlp_num_hidden_layers': 2,
        'num_message_passing_steps': 10
    }
    simulator = _get_simulator(model_kwargs, metadata, acc_noise_std=args.noise_std,
                               vel_noise_std=args.noise_std, device=device)

    if args.mode in ['train', 'eval']:
        # For one-step training/evaluation.
        mode_str = 'one_step_train' if args.mode == 'train' else 'one_step'
        split = 'train' if args.mode == 'train' else args.eval_split
        dataset = get_dataset(args.data_path, mode=mode_str, split=split, metadata=metadata)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(args.mode == 'train'),
                                collate_fn=reading_utils.collate_fn)
        if args.mode == 'train':
            optimizer = optim.Adam(simulator.parameters(), lr=1e-4)
            # (Optional) Define a learning rate scheduler.
            scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: (((1e-4 - 1e-6) * (0.1 ** (step / 5e6)) + 1e-6) / 1e-4)
            )
            global_step = train_one_step(simulator, dataloader, optimizer, args.noise_std,
                                         device, args.num_steps)
            scheduler.step()
            os.makedirs(args.model_path, exist_ok=True)
            torch.save(simulator.state_dict(), os.path.join(args.model_path, 'model.pth'))
        else:  # Evaluation.
            simulator.load_state_dict(torch.load(os.path.join(args.model_path, 'model.pth'),
                                                 map_location=device))
            evaluate_one_step(simulator, dataloader, args.noise_std, device)
    elif args.mode == 'eval_rollout':
        if args.output_path is None:
            raise ValueError("A rollout output path must be provided.")
        # For rollout evaluation, we use batch size 1.
        dataset = get_dataset(args.data_path, mode='rollout', split=args.eval_split, metadata=metadata)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=reading_utils.collate_fn)
        simulator.load_state_dict(torch.load(os.path.join(args.model_path, 'model.pth'),
                                             map_location=device))
        for example_index, features in enumerate(dataloader):
            # Determine the number of rollout steps.
            num_steps = metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
            rollout_result = rollout(simulator, features, num_steps, device)
            rollout_result['metadata'] = metadata
            filename = f'rollout_{args.eval_split}_{example_index}.pkl'
            os.makedirs(args.output_path, exist_ok=True)
            filepath = os.path.join(args.output_path, filename)
            logging.info(f"Saving rollout to: {filepath}")
            with open(filepath, 'wb') as f:
                pickle.dump(rollout_result, f)
    else:
        raise ValueError(f"Mode {args.mode} not recognized.")


if __name__ == '__main__':
    main()
