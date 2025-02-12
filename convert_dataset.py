#!/usr/bin/env python
import os
import argparse
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# These definitions match what is used in reading_utils.py.

_CONTEXT_FEATURES = {
    'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}

_FEATURE_DESCRIPTION = {
    'position': tf.io.VarLenFeature(tf.string),
}

_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(tf.string)

def parse_example(serialized_example, metadata):
    """
    Parses a single tf.SequenceExample into a dict with keys 'context' and 'feature_list'.
    
    Args:
      serialized_example: A serialized tf.SequenceExample.
      metadata: A dict with metadata (e.g. to decide whether to include global context).
    
    Returns:
      A dictionary with keys:
        - "context": dict of context features.
        - "feature_list": dict of sequence features.
    """
    if 'context_mean' in metadata:
        feature_description = _FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT
    else:
        feature_description = _FEATURE_DESCRIPTION

    context, feature_list = tf.io.parse_single_sequence_example(
        serialized_example,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description)
    return {'context': context, 'feature_list': feature_list}

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("Reading TFRecord file from: %s", os.path.join(args.data_path, f"{args.split}.tfrecord"))
    dataset = tf.data.TFRecordDataset(os.path.join(args.data_path, f"{args.split}.tfrecord"))
    # Map the parsing function.
    parsed_dataset = dataset.map(lambda ex: parse_example(ex, args.metadata))
    
    examples = []
    with tf.Session() as sess:
        iterator = parsed_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        try:
            while True:
                ex = sess.run(next_element)
                examples.append(ex)
        except tf.errors.OutOfRangeError:
            pass

    output_file = os.path.join(args.data_path, f"{args.split}.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(examples, f)
    print(f"Saved {len(examples)} examples to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TFRecord to Pickle for simulation data.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Directory containing the TFRecord file (e.g. /tmp/datasets/WaterRamps)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"],
                        help="Dataset split to convert (default: train)")
    # For simplicity, metadata can be passed via a file or hard-coded; here we use a simple string.
    # Adjust the metadata as needed (it must at least include 'sequence_length' and 'dim').
    parser.add_argument("--metadata", type=lambda s: eval(s), required=True,
                        help="Metadata dictionary (e.g. \"{'sequence_length': 6, 'dim': 3, 'context_mean': [0,0,0,0,0,0]}\" )")
    args = parser.parse_args()
    main(args)
