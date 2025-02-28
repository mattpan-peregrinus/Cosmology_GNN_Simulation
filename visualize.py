"""
convert_tfrecords.py

Example usage:
    python convert_tfrecords.py \
        --tfrecord_dir ./WaterRamps \
        --output_dir ./WaterRamps_converted \
        --dim 2
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf

def parse_example(example_proto):
    """
    Parse a single trajectory from a TFRecord.
    Adjust this to match the actual WaterRamps feature keys & shapes.
    """

    # WaterRamps often uses these keys (verify in the TFRecord):
    features = {
        'positions': tf.io.VarLenFeature(tf.float32),
        'particle_type': tf.io.VarLenFeature(tf.int64),
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    # Convert sparse to dense if VarLenFeature
    positions = tf.sparse.to_dense(parsed_features['positions'])       # shape = [time * n_particles * dim] or [time, n_particles, dim]
    particle_type = tf.sparse.to_dense(parsed_features['particle_type'])  # shape = [num_particles]

    # Depending on how WaterRamps stores positions, you may need to reshape here.
    # For demonstration, let's assume they are already [time, n_particles, dim].
    # If they're flattened, you'd do something like:
    #   total = positions.shape[0]  # e.g., time*n_particles*dim
    #   time = 600  # or read from metadata
    #   n_particles = ???  # depends on your dataset
    #   dim = 2
    #   positions = tf.reshape(positions, [time, n_particles, dim])

    return positions, particle_type


def convert_split(
    tfrecord_path,
    offset_json_path,
    position_dat_path,
    particle_type_dat_path,
    dim,
):
    """
    Reads all trajectories from one TFRecord (train/valid/test),
    writes them into .dat files, and creates an offset JSON dictionary.
    """

    position_file = open(position_dat_path, 'wb')
    type_file = open(particle_type_dat_path, 'wb')

    position_offset = 0
    type_offset = 0

    offsets_dict = {}
    trajectory_index = 0

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    for raw_example in dataset:
        positions_tf, ptypes_tf = parse_example(raw_example)
        positions_np = positions_tf.numpy()   # e.g. shape = [time, n_particles, dim]
        ptypes_np = ptypes_tf.numpy()         # shape = [num_particles]

        # If needed, reshape here. For example:
        # time, n_particles, real_dim = positions_np.shape
        # If it's already [time, n_particles, dim], great. Otherwise:
        # positions_np = positions_np.reshape([time, n_particles, dim])

        time, n_particles, real_dim = positions_np.shape

        # Write positions to .dat (float32)
        positions_np.astype(np.float32).tofile(position_file)
        current_position_offset = position_offset
        num_position_elements = positions_np.size
        position_offset += num_position_elements

        # Write particle types to .dat (int64)
        ptypes_np.astype(np.int64).tofile(type_file)
        current_type_offset = type_offset
        num_type_elements = ptypes_np.size
        type_offset += num_type_elements

        # Build the offset entry for this trajectory
        offsets_dict[trajectory_index] = {
            "particle_type": {
                "offset": current_type_offset
            },
            "position": {
                "offset": current_position_offset,
                "shape": [time, n_particles, real_dim]
            }
        }

        trajectory_index += 1

    position_file.close()
    type_file.close()

    with open(offset_json_path, 'w') as f:
        json.dump(offsets_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecord_dir', type=str, required=True,
                        help="Folder with train.tfrecord, valid.tfrecord, test.tfrecord, plus metadata.json (optional).")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Where to write .dat files, offset JSONs, and final metadata.json.")
    parser.add_argument('--dim', type=int, default=2,
                        help="Dimension of positions (2D or 3D). Only used if you need to reshape.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Convert train split
    train_record = os.path.join(args.tfrecord_dir, 'train.tfrecord')
    if os.path.exists(train_record):
        convert_split(
            tfrecord_path=train_record,
            offset_json_path=os.path.join(args.output_dir, 'train_offset.json'),
            position_dat_path=os.path.join(args.output_dir, 'train_position.dat'),
            particle_type_dat_path=os.path.join(args.output_dir, 'train_particle_type.dat'),
            dim=args.dim,
        )

    # Convert valid split
    valid_record = os.path.join(args.tfrecord_dir, 'valid.tfrecord')
    if os.path.exists(valid_record):
        convert_split(
            tfrecord_path=valid_record,
            offset_json_path=os.path.join(args.output_dir, 'valid_offset.json'),
            position_dat_path=os.path.join(args.output_dir, 'valid_position.dat'),
            particle_type_dat_path=os.path.join(args.output_dir, 'valid_particle_type.dat'),
            dim=args.dim,
        )

    # Convert test split
    test_record = os.path.join(args.tfrecord_dir, 'test.tfrecord')
    if os.path.exists(test_record):
        convert_split(
            tfrecord_path=test_record,
            offset_json_path=os.path.join(args.output_dir, 'test_offset.json'),
            position_dat_path=os.path.join(args.output_dir, 'test_position.dat'),
            particle_type_dat_path=os.path.join(args.output_dir, 'test_particle_type.dat'),
            dim=args.dim,
        )

    # -- METADATA HANDLING --

    # If a metadata.json already exists in tfrecord_dir, copy it over.
    # Otherwise, we'll create the exact WaterRamps metadata you provided.
    src_meta = os.path.join(args.tfrecord_dir, 'metadata.json')
    dst_meta = os.path.join(args.output_dir, 'metadata.json')
    if os.path.exists(src_meta):
        import shutil
        shutil.copyfile(src_meta, dst_meta)
    else:
        # Use the WaterRamps metadata you gave:
        meta = {
            "bounds": [[0.1, 0.9], [0.1, 0.9]],
            "sequence_length": 600,
            "default_connectivity_radius": 0.015,
            "dim": 2,
            "dt": 0.0025,
            "vel_mean": [-6.141567458658365e-08, -0.0007425391691160353],
            "vel_std": [0.0022381126134429557, 0.0022664486850394443],
            "acc_mean": [-1.713503820317499e-07, -2.1448168008479274e-07],
            "acc_std": [0.00016824548701156486, 0.0001819676291787043]
        }
        with open(dst_meta, 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"Conversion complete! Output files in {args.output_dir}")


if __name__ == '__main__':
    main()

