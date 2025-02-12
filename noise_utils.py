import torch
import learned_simulator

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    """
    Returns random-walk noise in the velocity applied to the position.
    
    This function first computes the velocity differences (using a finite-difference
    approximation) from the input position sequence. It then generates per-timestep 
    noise (with a standard deviation scaled so that, when composed as a random walk, 
    the noise at the final step has a fixed standard deviation). The noise is first
    accumulated (cumsum) to produce a random-walk in velocity, and then integrated
    (cumsum again) to yield a noise trajectory for the positions. A zero is prepended 
    so that the initial position remains unaltered.
    
    Args:
      position_sequence: torch.Tensor of shape [num_particles, sequence_length, num_dimensions].
                         The first time step is assumed to be noise-free.
      noise_std_last_step: float, desired standard deviation of the noise in the velocity 
                           at the final step.
    
    Returns:
      position_sequence_noise: torch.Tensor of shape [num_particles, sequence_length, num_dimensions],
                               containing the integrated noise that can be added to the original
                               positions.
    """
    # Compute velocity differences; shape will be [num_particles, sequence_length - 1, num_dimensions].
    velocity_sequence = learned_simulator.time_diff(position_sequence)
    
    if velocity_sequence.shape[1] < 1:
      raise ValueError(f"Input position sequence must have at least 2 timesteps. Got shape: {position_sequence.shape}")
    
    # Determine the number of velocity timesteps.
    num_velocities = velocity_sequence.shape[1]
    
    # Compute the noise standard deviation per timestep. This ensures that when the noise is
    # accumulated over all timesteps, the standard deviation at the final step is fixed.
    noise_std_each = noise_std_last_step / (num_velocities ** 0.5)
    
    # Sample random noise for the velocity (with the appropriate standard deviation).
    velocity_sequence_noise = torch.randn_like(velocity_sequence) * noise_std_each
    
    # Generate a random-walk in velocity by cumulatively summing along the time dimension.
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    
    # Integrate the velocity noise to obtain position noise.
    # Prepend a zero tensor so that the very first position remains unaltered.
    integrated_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    zeros = torch.zeros_like(integrated_noise[:, :1])
    position_sequence_noise = torch.cat([zeros, integrated_noise], dim=1)
    
    return position_sequence_noise
