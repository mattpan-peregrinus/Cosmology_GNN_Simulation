#!/usr/bin/env python
"""
Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):
  python render_rollout.py --rollout_path {OUTPUT_PATH}/rollout_test_1.pkl

It may require installing Tkinter (e.g. `sudo apt-get install python3-tk`).
"""

import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Define a mapping from particle type to color.
TYPE_TO_COLOR = {
    3: "black",   # Boundary particles.
    0: "green",   # Rigid solids.
    7: "magenta", # Goop.
    6: "gold",    # Sand.
    5: "blue",    # Water.
}


def main(args):
    if not args.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")

    # Load rollout data from the pickle file.
    with open(args.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    # Create a figure with two subplots: one for ground truth and one for prediction.
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_info = []

    # For each subplot, set up the title, bounds, and create empty plot lines for each particle type.
    for ax_i, (label, rollout_field) in enumerate(
            [("Ground truth", "ground_truth_rollout"),
             ("Prediction", "predicted_rollout")]):
        # Concatenate the initial positions with the rollout trajectory.
        trajectory = np.concatenate([
            rollout_data["initial_positions"],
            rollout_data[rollout_field]
        ], axis=0)
        ax = axes[ax_i]
        ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]
        # Assumes the first two dimensions correspond to x and y.
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.)
        # Create an empty plot (line object) for each particle type.
        points = {
            particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
            for particle_type, color in TYPE_TO_COLOR.items()
        }
        plot_info.append((ax, trajectory, points))

    # Determine the number of steps in the trajectory.
    num_steps = trajectory.shape[0]

    def update(step_i):
        outputs = []
        for _, trajectory, points in plot_info:
            for particle_type, line in points.items():
                # Create a boolean mask for particles of this type.
                mask = rollout_data["particle_types"] == particle_type
                # Update the data for this particle type.
                line.set_data(trajectory[step_i, mask, 0],
                              trajectory[step_i, mask, 1])
                outputs.append(line)
        return outputs

    # Create an animation that calls `update` for each frame.
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=np.arange(0, num_steps, args.step_stride),
        interval=10
    )

    plt.show(block=args.block_on_show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render rollout prediction against ground truth.")
    parser.add_argument(
        "--rollout_path",
        type=str,
        required=True,
        help="Path to rollout pickle file")
    parser.add_argument(
        "--step_stride",
        type=int,
        default=3,
        help="Stride of steps to skip.")
    parser.add_argument(
        "--block_on_show",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Block on plt.show? (True or False)")
    args = parser.parse_args()
    main(args)
