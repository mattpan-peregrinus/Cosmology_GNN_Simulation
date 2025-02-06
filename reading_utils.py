import functools
import numpy as np
import torch

# --- Definitions that mirror the original TF feature specs. ---

# In the TF version, positions (and optionally step_context) are stored as VarLenFeature of tf.string.
# Here we assume that in each parsed example these are lists of bytes.
_FEATURE_DESCRIPTION = {
    'position': 'byte',
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = 'byte'

# Define the expected input dtype for each feature.
_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,   # raw bytes represent float32 numbers
        'out': torch.float32
    },
    'step_context': {
        'in': np.float32,
        'out': torch.float32
    }
}

# Context features (non-sequence features)
_CONTEXT_FEATURES = {
    'key': 'int',
    'particle_type': 'byte'  # Stored as bytes; will be decoded into int64.
}


# --- Utility functions ---

def convert_to_tensor(x, encoded_dtype):
    """
    Converts a list of bytes objects (each storing a binary buffer) into a torch.Tensor.
    
    Args:
      x: A list of bytes objects.
      encoded_dtype: A NumPy dtype (e.g. np.float32 or np.int64) indicating how to interpret the bytes.
    
    Returns:
      A torch.Tensor containing the decoded data.
    """
    if isinstance(x, list) and len(x) == 1:
        # If a single bytes object is provided, decode its entire buffer.
        out = np.frombuffer(x[0], dtype=encoded_dtype)
    else:
        out = [np.frombuffer(el, dtype=encoded_dtype) for el in x]
        out = np.array(out)
    return torch.tensor(out)


def parse_serialized_simulation_example(record, metadata):
    """
    Parses a serialized simulation example (originally a tf.SequenceExample) into context and sequence features.
    
    Args:
      record: A dict with keys:
              - 'context': a dict of context features (e.g. 'key', 'particle_type'),
              - 'feature_list': a dict of sequence features (e.g. 'position', optionally 'step_context'),
                where each value is a list of bytes objects.
      metadata: A dict of metadata (must include 'sequence_length' and 'dim', and optionally 'context_mean').
    
    Returns:
      A tuple (context, parsed_features) where:
         - context is a dict containing context features (with 'particle_type' decoded to int64),
         - parsed_features is a dict of torch.Tensors for each sequence feature.
           In particular, 'position' is reshaped to [sequence_length+1, num_particles, dim];
           if present, 'step_context' is reshaped to [sequence_length+1, context_feat_len].
    """
    # Choose which feature description to use based on metadata.
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION

    # Assume the record is already split into context and feature_list.
    context = record.get('context', {})
    parsed_features = record.get('feature_list', {})

    # Convert each sequence feature from raw bytes to a tensor.
    for feature_key, value in parsed_features.items():
        convert_fn = functools.partial(convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in'])
        parsed_features[feature_key] = convert_fn(value)
    
    # Reshape the position tensor.
    # Expected shape: [metadata['sequence_length'] + 1, num_particles, metadata['dim']]
    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]
    parsed_features['position'] = parsed_features['position'].reshape(position_shape)
    
    # If step_context is present, reshape it.
    sequence_length = metadata['sequence_length'] + 1
    if 'context_mean' in metadata and 'step_context' in parsed_features:
        context_feat_len = len(metadata['context_mean'])
        parsed_features['step_context'] = parsed_features['step_context'].reshape([sequence_length, context_feat_len])
    
    # Decode particle_type explicitly.
    if 'particle_type' in context:
        pt_list = context['particle_type']
        # Assume pt_list is a list of bytes objects; convert to a NumPy array of int64.
        if isinstance(pt_list, list) and len(pt_list) == 1:
            particle_type_np = np.frombuffer(pt_list[0], dtype=np.int64)
        else:
            particle_type_np = np.array([np.frombuffer(el, dtype=np.int64) for el in pt_list])
        context['particle_type'] = torch.tensor(particle_type_np, dtype=torch.long).reshape(-1)
    
    return context, parsed_features


def split_trajectory(context, features, window_length=7):
    """
    Splits a long trajectory into overlapping sliding windows.
    
    For example, if the original position tensor has shape
       [trajectory_length, num_particles, dim],
    then each window will be of shape [window_length, num_particles, dim]
    and the number of windows is (trajectory_length - window_length + 1).
    
    Additionally, the particle types (from context) are tiled appropriately.
    If available, step_context is also split into sliding windows.
    
    Args:
      context: A dict containing context features (e.g. 'particle_type').
      features: A dict of sequence features, which must include 'position' (a torch.Tensor)
                of shape [trajectory_length, num_particles, dim]. Optionally, it may include
                'step_context' of shape [trajectory_length, context_feat_len].
      window_length: An integer for the length of each sliding window.
    
    Returns:
      A list of dictionaries. Each dictionary represents one sliding window and contains:
          - 'position': Tensor of shape [window_length, num_particles, dim]
          - 'particle_type': Tensor of shape [num_particles] (tiled from context)
          - (optionally) 'step_context': Tensor of shape [window_length, context_feat_len]
    """
    trajectory_length = features['position'].shape[0]
    input_trajectory_length = trajectory_length - window_length + 1
    examples = []
    
    # Extract particle_type from context (assumed constant over time).
    particle_type = context.get('particle_type')
    
    for idx in range(input_trajectory_length):
        example = {}
        example['position'] = features['position'][idx: idx + window_length]
        if 'step_context' in features:
            example['step_context'] = features['step_context'][idx: idx + window_length]
        if particle_type is not None:
            example['particle_type'] = particle_type
        examples.append(example)
    return examples
