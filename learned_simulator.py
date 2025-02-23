import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our converted graph network.
import graph_network
# Import connectivity utilities.
import connectivity_utils

# A small epsilon for numerical stability.
STD_EPSILON = 1e-8


class LearnedSimulator(nn.Module):
    """
    Learned simulator based on "Learning to Simulate Complex Physics with Graph Networks"
    (https://arxiv.org/abs/2002.09405). This module predicts one simulation step.
    """
    def __init__(self,
                 num_dimensions,
                 connectivity_radius,
                 graph_network_kwargs,
                 boundaries,
                 normalization_stats,
                 num_particle_types,
                 particle_type_embedding_size,
                 name="LearnedSimulator"):
        """
        Args:
          num_dimensions: Dimensionality of the simulation (e.g. 2 or 3).
          connectivity_radius: Scalar radius used for graph connectivity.
          graph_network_kwargs: Dictionary of keyword arguments to pass to the
              EncodeProcessDecode module.
          boundaries: List/array of shape [num_dimensions, 2] containing the lower
              and upper boundaries along each dimension.
          normalization_stats: Dictionary with keys "acceleration", "velocity",
              (and optionally "context") whose values have attributes `mean` and `std`.
          num_particle_types: Number of different particle types.
          particle_type_embedding_size: Size of the particle type embedding.
          name: (Unused) name of the module.
        """
        super().__init__()
        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats

        # Build the learned graph network. (Our converted EncodeProcessDecode is a PyTorch module.)
        self._graph_network = graph_network.EncodeProcessDecode(
            output_size=num_dimensions, **graph_network_kwargs)

        # Use an embedding layer if there are multiple particle types.
        if self._num_particle_types > 1:
            self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size)

    def forward(self, position_sequence, n_particles_per_example,
                global_context=None, particle_types=None):
        """
        Runs one simulation step.

        Args:
          position_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
          n_particles_per_example: Number (or tensor/list) of particles per graph (batch element).
          global_context: Tensor of shape [batch_size, context_size] (optional).
          particle_types: Tensor of shape [num_particles] with integer types (optional).

        Returns:
          next_position: Tensor of shape [num_particles, num_dimensions] predicting the next positions.
        """
        # Build the input graph from the positions and extra features.
        input_graph = self._encoder_preprocessor(position_sequence, n_particles_per_example,
                                                   global_context, particle_types)
        # Predict normalized acceleration via the graph network.
        normalized_acceleration = self._graph_network(input_graph)
        # Convert acceleration back to positions using a simple Euler update.
        next_position = self._decoder_postprocessor(normalized_acceleration, position_sequence)
        return next_position

    def _encoder_preprocessor(self, position_sequence, n_node, global_context, particle_types):
        """
        Constructs a graph from the position sequence for GraphSAGE processing.
        Focuses on building rich node features since GraphSAGE primarily uses node information.
    
        Args:
          position_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
          n_node: Number (or tensor/list) of nodes per graph in the batch.
          global_context: Tensor of shape [batch_size, context_size] (optional).
          particle_types: Tensor of shape [num_particles] with integer types (optional).
    
        Returns:
          A torch_geometric.data.Data object containing:
            - x: concatenated node features
            - edge_index: [2, num_edges] connectivity
            - globals (optional): normalized global context
        """
        # Get the most recent positions.
        most_recent_position = position_sequence[:, -1]  # [num_particles, num_dimensions]
        # Compute finite differences to approximate velocity.
        velocity_sequence = time_diff(position_sequence)  # [num_particles, sequence_length-1, num_dimensions]

        # Compute graph connectivity using your (converted) connectivity_utils.
        # Expected to return tensors: senders, receivers, and n_edge.
        senders, receivers, n_edge = connectivity_utils.compute_connectivity_for_batch_pyfunc(
            most_recent_position, n_node, self._connectivity_radius)

        # Build node features.
        node_features = []
        
        # 1. Position features.
        node_features.append(most_recent_position)
        
        # 2. Velocity features.
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (velocity_sequence - velocity_stats.mean) / velocity_stats.std
        flat_velocity_sequence = normalized_velocity_sequence.reshape(normalized_velocity_sequence.size(0), -1)
        node_features.append(flat_velocity_sequence)
        
        # 3. Boundary features.
        boundaries = torch.tensor(self._boundaries, dtype=most_recent_position.dtype,
                              device=most_recent_position.device)
        distance_to_lower_boundary = most_recent_position - boundaries[:, 0].unsqueeze(0)
        distance_to_upper_boundary = boundaries[:, 1].unsqueeze(0) - most_recent_position
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=-1)
        normalized_clipped_distance_to_boundaries = torch.clamp(
            distance_to_boundaries / self._connectivity_radius, -1., 1.)
        node_features.append(normalized_clipped_distance_to_boundaries)
        
        # 4. Particle type embeddings. 
        if self._num_particle_types > 1 and particle_types is not None:
          particle_type_embeddings = self._particle_type_embedding(particle_types)
          node_features.append(particle_type_embeddings)
          
        # 5. Optional: Add more node features here
        # For example:
        # - Local density features
        # - Historical movement patterns
        # - Aggregated neighborhood statistics
        # TODO: Add additional node features as needed

        # Concatenate all node features.
        nodes = torch.cat(node_features, dim=-1)

        # Normalize global context if provided.
        globals_val = None
        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            global_context = (global_context - context_stats.mean) / torch.max(
                context_stats.std, torch.tensor(STD_EPSILON, device=global_context.device, dtype=global_context.dtype))
            globals_val = global_context

        # Build a PyTorch Geometric Data object.
        from torch_geometric.data import Data
        edge_index = torch.stack([senders, receivers], dim=0)
        # Note: We don't include edge_attr since GraphSAGE doesn't use it
        data = Data(x=nodes, edge_index=edge_index)
        if globals_val is not None:
            data.globals = globals_val
        data.n_node = n_node
        data.n_edge = n_edge
        return data
    
    
    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        """
        Converts the model's output from normalized acceleration space back to positions.
        Uses an Euler integrator with a time step of 1.
        
        Args:
          normalized_acceleration: Tensor of shape [num_particles, num_dimensions]
          position_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
        
        Returns:
          new_position: Tensor of shape [num_particles, num_dimensions]
        """
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = normalized_acceleration * acceleration_stats.std + acceleration_stats.mean
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]
        new_velocity = most_recent_velocity + acceleration  # dt assumed to be 1.
        new_position = most_recent_position + new_velocity
        return new_position

    def get_predicted_and_target_normalized_accelerations(self,
                                                          next_position,
                                                          position_sequence_noise,
                                                          position_sequence,
                                                          n_particles_per_example,
                                                          global_context=None,
                                                          particle_types=None):
        """
        Computes both the predicted normalized acceleration (via the noisy input)
        and the target normalized acceleration (via an inverse decoder update).

        Args:
          next_position: Tensor of shape [num_particles, num_dimensions] (ground truth next positions)
          position_sequence_noise: Noise tensor of the same shape as position_sequence.
          position_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
          n_particles_per_example: Number (or tensor/list) of particles per graph.
          global_context: Tensor of shape [batch_size, context_size] (optional).
          particle_types: Tensor of shape [num_particles] with particle types (optional).

        Returns:
          A tuple (predicted_normalized_acceleration, target_normalized_acceleration)
        """
        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Forward pass with noisy input.
        input_graph = self._encoder_preprocessor(noisy_position_sequence, n_particles_per_example,
                                                   global_context, particle_types)
        predicted_normalized_acceleration = self._graph_network(input_graph)

        # Adjust the next position by adding the noise at the last time step.
        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence)
        return predicted_normalized_acceleration, target_normalized_acceleration

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        """
        Computes the normalized acceleration target given the next position and
        the input position sequence. This is the inverse of the Euler update.
        
        Args:
          next_position: Tensor of shape [num_particles, num_dimensions]
          position_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
        
        Returns:
          normalized_acceleration: Tensor of shape [num_particles, num_dimensions]
        """
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (acceleration - acceleration_stats.mean) / acceleration_stats.std
        return normalized_acceleration


def time_diff(input_sequence):
    """
    Computes finite differences along the time dimension.
    
    Args:
      input_sequence: Tensor of shape [num_particles, sequence_length, num_dimensions]
    
    Returns:
      Tensor of shape [num_particles, sequence_length - 1, num_dimensions]
    """
    return input_sequence[:, 1:] - input_sequence[:, :-1]
