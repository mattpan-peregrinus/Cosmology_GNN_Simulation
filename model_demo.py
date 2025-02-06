import collections
import numpy as np
import torch

import learned_simulator
import noise_utils

# Constants.
INPUT_SEQUENCE_LENGTH = 6
SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH + 1  # input + target
NUM_DIMENSIONS = 3
NUM_PARTICLE_TYPES = 6
BATCH_SIZE = 5
GLOBAL_CONTEXT_SIZE = 6

# Dummy statistics and boundaries for normalization.
Stats = collections.namedtuple("Stats", ["mean", "std"])

DUMMY_STATS = Stats(
    mean=np.zeros([NUM_DIMENSIONS], dtype=np.float32),
    std=np.ones([NUM_DIMENSIONS], dtype=np.float32)
)
DUMMY_CONTEXT_STATS = Stats(
    mean=np.zeros([GLOBAL_CONTEXT_SIZE], dtype=np.float32),
    std=np.ones([GLOBAL_CONTEXT_SIZE], dtype=np.float32)
)
DUMMY_BOUNDARIES = [(-1.0, 1.0)] * NUM_DIMENSIONS


def sample_random_position_sequence():
    """
    Returns mock data mimicking the input features collected by the encoder.
    Produces a tensor of shape [num_particles, SEQUENCE_LENGTH, NUM_DIMENSIONS],
    with a random number of particles between 50 and 1000.
    """
    num_particles = torch.randint(low=50, high=1000, size=(1,)).item()
    position_sequence = torch.randn(num_particles, SEQUENCE_LENGTH, NUM_DIMENSIONS)
    return position_sequence


def main():
    # Build the model.
    learnable_model = learned_simulator.LearnedSimulator(
        num_dimensions=NUM_DIMENSIONS,
        connectivity_radius=0.05,
        graph_network_kwargs={
            'latent_size': 128,
            'mlp_hidden_size': 128,
            'mlp_num_hidden_layers': 2,
            'num_message_passing_steps': 10,
        },
        boundaries=DUMMY_BOUNDARIES,
        normalization_stats={
            "acceleration": DUMMY_STATS,
            "velocity": DUMMY_STATS,
            "context": DUMMY_CONTEXT_STATS,
        },
        num_particle_types=NUM_PARTICLE_TYPES,
        particle_type_embedding_size=16,
    )
    # Set model to evaluation mode.
    learnable_model.eval()

    # Sample a batch of particle sequences.
    sampled_position_sequences = [sample_random_position_sequence() for _ in range(BATCH_SIZE)]
    # Concatenate along the particle axis to form a single tensor.
    position_sequence_batch = torch.cat(sampled_position_sequences, dim=0)

    # Compute the number of particles in each example.
    n_particles_per_example = torch.tensor(
        [seq.shape[0] for seq in sampled_position_sequences], dtype=torch.long)

    # Total number of particles in the batch.
    total_particles = position_sequence_batch.shape[0]

    # Sample particle types: a random integer in [0, NUM_PARTICLE_TYPES) for each particle.
    particle_types = torch.randint(0, NUM_PARTICLE_TYPES, (total_particles,), dtype=torch.long)

    # Sample global context for each example; values in [-1, 1].
    global_context = 2 * torch.rand(BATCH_SIZE, GLOBAL_CONTEXT_SIZE) - 1

    # Separate the input sequence (first SEQUENCE_LENGTH - 1 steps) and target (last step).
    input_position_sequence = position_sequence_batch[:, :-1, :]  # [total_particles, INPUT_SEQUENCE_LENGTH, NUM_DIMENSIONS]
    target_next_position = position_sequence_batch[:, -1, :]      # [total_particles, NUM_DIMENSIONS]

    # --- Inference ---
    # Run a single step of inference to predict next positions.
    with torch.no_grad():
        predicted_next_position = learnable_model(
            input_position_sequence,
            n_particles_per_example,
            global_context,
            particle_types
        )
    print("Per-particle output tensor:")
    print(predicted_next_position)

    # --- Training Mode Example ---
    # Get noise for the input position sequence.
    position_sequence_noise = noise_utils.get_random_walk_noise_for_position_sequence(
        input_position_sequence, noise_std_last_step=6.7e-4)

    # Obtain predicted and target normalized accelerations.
    with torch.no_grad():
        predicted_norm_acc, target_norm_acc = learnable_model.get_predicted_and_target_normalized_accelerations(
            target_next_position,
            position_sequence_noise,
            input_position_sequence,
            n_particles_per_example,
            global_context,
            particle_types
        )
    print("Predicted normalized acceleration:")
    print(predicted_norm_acc)
    print("Target normalized acceleration:")
    print(target_norm_acc)


if __name__ == "__main__":
    main()
