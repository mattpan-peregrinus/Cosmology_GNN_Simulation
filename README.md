# Cosmological Physics Simulation with GNNs

Learning cosmological particle dynamics using Graph Neural Networks with physics-informed constraints.

## Overview

This system predicts particle accelerations and temperature changes by:
- Building graphs where particles are nodes connected to nearby neighbors
- Using message passing to simulate particle interactions
- Enforcing conservation of momentum through physics-informed loss terms

## Architecture

**Data Processing** - Handles periodic boundaries, constructs k-NN graphs, normalizes features

**Graph Network** - Interaction Network with custom message passing and edge updates

**Training Pipeline** - Multi-objective loss (acceleration + temperature + momentum conservation)

## Requirements

```bash
bash setup_env.sh
```

Key dependencies: PyTorch 2.6.0+, PyTorch Geometric 2.6.1+, HDF5 support

## Quick Start

### Data Format
HDF5 files with: `Coordinates`, `InternalEnergy`, `Velocities`, `HydroAcceleration`, `BoxSize`, `TimeStep`

### Generate Metadata
```bash
python generate_metadata.py --dataset path/to/data.hdf5 --output metadata.json
```

### Train
```bash
python train.py \
  --train_dir path/to/training/data \
  --val_dir path/to/validation/data \
  --metadata_path metadata.json \
  --output_dir model_output \
  --num_epochs 50
```

### Evaluate
```bash
# One-step test
python one_step_test.py --model_path model_output/model_best.pth --test_data path/to/test.hdf5

# Multi-step rollout
python render_rollout.py --model_path model_output/model_best.pth --test_data path/to/test.hdf5
```

## Key Parameters

- `latent_size`: Representation size (64, 128, 256)
- `num_message_passing_steps`: Message passing rounds (5-15)
- `num_neighbors`: Graph connectivity (8-32)
- `learning_rate`: 1e-4 → 1e-5 with decay

## Project Structure

```
├── train.py                 # Training script
├── one_step_test.py        # Single timestep test
├── render_rollout.py       # Multi-timestep simulation
├── data_utils.py           # Data preprocessing and graphs
├── graph_network.py        # Interaction Network
├── generate_metadata.py   # Normalization statistics
└── setup_env.sh          # Environment setup
```

## Contributors

Developed under Carnegie Mellon's McWilliams Center for Cosmology and Astrophysics.

## References

- [Graph Neural Networks for Physics Simulations](https://arxiv.org/abs/2002.09405)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
