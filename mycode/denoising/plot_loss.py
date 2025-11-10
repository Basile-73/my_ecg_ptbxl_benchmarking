#!/usr/bin/env python3
"""
Script to plot training and validation loss from a history JSON file.

Usage:
    python plot_loss.py <path_to_history.json>

Example:
    python plot_loss.py output/test_mamba_models_experiment/models/imunet_mamba_late/history.json
"""

import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(history_path):
    """
    Plot training and validation loss from a history JSON file.

    Args:
        history_path (str): Path to the history.json file
    """
    # Load the history file
    with open(history_path, 'r') as f:
        history = json.load(f)

    train_loss = history['train_loss']
    val_loss = history['val_loss']

    # Create epochs array
    epochs = np.arange(1, len(train_loss) + 1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', linewidth=2, marker='o', markersize=5)
    plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=5)

    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=22, fontweight='bold')
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()

    output_dir = os.path.dirname(history_path)
    output_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    output_path_pdf = os.path.join(output_dir, 'loss_plot.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"Plot saved to: {output_path_pdf}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot training and validation loss from a history JSON file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python plot_loss.py output/test_mamba_models_experiment/models/imunet_mamba_late/history.json
        """
    )
    parser.add_argument(
        'history_path',
        type=str,
        help='Path to the history.json file'
    )

    args = parser.parse_args()

    if not os.path.exists(args.history_path):
        print(f"Error: File not found: {args.history_path}")
        return 1

    if not args.history_path.endswith('.json'):
        print(f"Warning: The file does not have a .json extension: {args.history_path}")

    try:
        plot_loss(args.history_path)
        print("Done!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
