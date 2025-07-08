# Cosmological Physics Simulation with GNNs

Learning cosmological particle dynamics using Graph Neural Networks (GNNs). This system implements an Encode-Process-Decode architecture to predict both particle accelerations and temperature changes in physical simulations.

## Overview

This system learns to simulate particle physics by:
- Taking a sequence of particle positions and temperatures as input
- Building a graph where particles are nodes and nearby particles are connected by edges
- Using a Graph Neural Network to predict how particles will move and heat up/cool down
- Supporting both GraphSAGE and Interaction Network architectures

## Architecture

### Core Components

**Data Processing** 
- Handles periodic boundary conditions for simulation boxes
- Constructs k-nearest neighbor graphs between particles
- Normalizes features using training statistics

**Graph Networks** 
- **Interaction Network**: Custom message passing with explicit edge updates
- **GraphSAGE**: Uses PyTorch Geometric's SAGEConv with max aggregation
- Both follow Encode-Process-Decode pattern

**Training Pipeline** 
- Multi-objective loss combining acceleration and temperature prediction
- Data augmentation (time reversal, axis permutation)
- Learning rate scheduling with exponential decay

### Why Graph Neural Networks?

Graph Neural Networks are ideal for particle simulation because:
- Particles interact primarily with nearby neighbors
- Graph structure naturally captures these local interactions
- Message passing allows information to propagate between connected particles
- More efficient than computing all-to-all particle interactions

### The Encode-Process-Decode Pattern

1. **Encode**: Transform raw particle features into latent representations
2. **Process**: Multiple rounds of message passing to simulate interactions
3. **Decode**: Transform latent features back to physical predictions

## Requirements

### Environment Setup
```bash
bash setup_env.sh
```

### Key Dependencies
- PyTorch 2.6.0+ with CUDA support
- PyTorch Geometric 2.6.1+
- HDF5 support (`h5py`)
- Standard scientific Python stack

## Quick Start

### 1. Prepare Your Data
Your simulation data should be in HDF5 format with these fields:
- `Coordinates`: Particle positions `[time_steps, num_particles, 3]`
- `InternalEnergy`: Particle temperatures `[time_steps, num_particles, 1]`
- `Velocities`: Particle velocities `[time_steps, num_particles, 3]`
- `HydroAcceleration`: Particle accelerations `[time_steps, num_particles, 3]`
- `BoxSize`: Simulation box size (scalar)
- `TimeStep`: Time step size (scalar)

### 2. Generate Metadata
```bash
python generate_metadata.py --dataset path/to/your/data.hdf5 --output metadata.json
```

### 3. Train the Model
```bash
python train.py \
  --train_dir path/to/training/data \
  --val_dir path/to/validation/data \
  --metadata_path metadata.json \
  --output_dir model_output \
  --num_epochs 50 
```

### 4. Evaluate the Model

**One-Step Prediction Test**
```bash
python one_step_test.py \
  --model_path model_output/model_best.pth \
  --test_data path/to/test/data.hdf5 \
  --metadata_path metadata.json \
  --num_timesteps 50
```

**Multi-Step Rollout**
```bash
python render_rollout.py \
  --model_path model_output/model_best.pth \
  --test_data path/to/test/data.hdf5 \
  --metadata_path metadata.json \
  --output_dir rollout_results
```

## Key Parameters

**Architecture:**
- `latent_size`: Size of learned representations (64, 128, 256)
- `num_message_passing_steps`: Rounds of message passing (5-15)
- `num_neighbors`: Particles connected to each node (8-32)

**Training:**
- `window_size`: Input timesteps to predict next timestep (default: 5)
- `learning_rate`: Start with 1e-4, decay to 1e-5
- `noise_std`: Training noise for robustness (0.0001-0.001)

## Project Structure

```
├── train.py                 # Main training script
├── validation.py           # Validation during training
├── one_step_test.py        # Single timestep accuracy test
├── render_rollout.py       # Multi-timestep simulation
├── data_utils.py           # Data preprocessing and graph construction
├── dataloader.py           # PyTorch dataset for loading sequences
├── graph_network.py        # Interaction Network implementation
├── config.py              # Command line argument parsing
├── generate_metadata.py   # Compute normalization statistics
├── rollout_conversion.py  # Convert rollout results to HDF5
└── setup_env.sh          # Environment setup script
```

## Training Outputs

The training script generates:
- **Loss plots**: Track training/validation loss over time
- **Component losses**: Separate tracking for acceleration vs temperature loss
- **Learning rate schedule**: Exponential decay visualization
- **Model checkpoints**: Best model and periodic saves

## Troubleshooting

**Common Issues:**
1. **CUDA out of memory**: Reduce batch_size or latent_size
2. **Poor convergence**: Check data normalization, try lower learning rate
3. **Unstable rollouts**: Increase noise_std during training
4. **Slow training**: Reduce num_neighbors or num_message_passing_steps

## Contributors

This project is developed under Carnegie Mellon's McWilliams Center and Cosmology and Astrophysics. 

## Further Reading

- [Graph Neural Networks for Physics Simulations](https://arxiv.org/abs/2002.09405)
- [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
