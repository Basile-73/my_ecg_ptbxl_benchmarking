"""
Plot Training Data Size Experiment Loss Curves

This script generates comprehensive visualizations of training dynamics across
different training set sizes, showing how models converge with varying amounts
of training data.

Features:
- Converts epoch-based metrics to step-based for fair comparison
- Creates per-model plots showing scaling across fold sizes
- Creates per-fold plots comparing models at each training size
- Generates comprehensive grid overview of all experiments
- Produces convergence analysis and learning rate schedules
- Exports quantitative summary statistics
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict

# Set plot style for publication quality
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def load_experiment_metadata(experiment_folder):
    """
    Load and validate experiment metadata.

    Args:
        experiment_folder: Path to training_data_size_experiment folder

    Returns:
        Dictionary with metadata
    """
    print("\n" + "="*80)
    print("LOADING EXPERIMENT METADATA")
    print("="*80)

    metadata_path = os.path.join(experiment_folder, 'experiment_metadata.json')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please ensure the experiment has completed successfully."
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Validate required fields
    required_fields = ['folds_tested', 'models_trained', 'fold_experiments']
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Required field '{field}' missing from metadata")

    # Print summary
    print(f"✓ Metadata loaded successfully")
    print(f"  Experiment date: {metadata.get('experiment_date', 'N/A')}")
    print(f"  Number of folds: {len(metadata['folds_tested'])}")
    print(f"  Number of models: {len(metadata['models_trained'])}")
    print(f"  Folds tested: {metadata['folds_tested']}")
    print(f"  Models trained: {metadata['models_trained']}")

    return metadata


def load_history_with_steps(history_path, steps_per_epoch, epochs_used):
    """
    Load a single model's history and convert to step-based metrics.

    Args:
        history_path: Path to history.json file
        steps_per_epoch: Number of training steps per epoch
        epochs_used: Expected number of epochs (may differ from actual if early stopping)

    Returns:
        Dictionary with step-based metrics, or None if file not found
    """
    if not os.path.exists(history_path):
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    # Validate expected fields
    required_fields = ['train_loss', 'val_loss']
    for field in required_fields:
        if field not in history:
            print(f"  ⚠️  Warning: '{field}' missing from {history_path}")
            return None

    # Get actual epochs trained (may be less than epochs_used due to early stopping)
    actual_epochs = len(history['train_loss'])

    if actual_epochs == 0:
        print(f"  ⚠️  Warning: Empty history in {history_path}")
        return None

    # Convert to numpy arrays
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    lr = np.array(history.get('lr', [0.001] * actual_epochs))  # Default LR if not recorded

    # Calculate cumulative steps at END of each epoch
    # For epoch i (0-indexed), steps = (i + 1) * steps_per_epoch
    steps = np.array([(i + 1) * steps_per_epoch for i in range(actual_epochs)])

    # Create epoch numbers (1-indexed for display)
    epochs = np.arange(1, actual_epochs + 1)

    return {
        'steps': steps,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'lr': lr,
        'epochs': epochs,
        'actual_epochs': actual_epochs,
        'expected_epochs': epochs_used
    }


def load_all_histories(metadata):
    """
    Load all history files across all folds and models.

    Args:
        metadata: Experiment metadata dictionary

    Returns:
        Nested dictionary: histories[model_name][fold_size] = history_dict
    """
    print("\n" + "="*80)
    print("LOADING TRAINING HISTORIES")
    print("="*80)

    histories = defaultdict(dict)
    total_loaded = 0
    total_missing = 0

    for fold in metadata['folds_tested']:
        fold_str = str(fold)

        if fold_str not in metadata['fold_experiments']:
            print(f"\n⚠️  Fold {fold} not found in metadata, skipping...")
            continue

        fold_info = metadata['fold_experiments'][fold_str]
        fold_experiment_path = fold_info.get('experiment_folder', '')

        # Normalize to absolute path to avoid cwd-dependent resolution
        fold_experiment_path = os.path.abspath(fold_experiment_path)

        if not os.path.exists(fold_experiment_path):
            print(f"\n⚠️  Fold experiment path not found: {fold_experiment_path}")
            continue

        print(f"\n--- Loading Fold {fold} ---")

        models_info = fold_info.get('models', {})

        for model_name in metadata['models_trained']:
            if model_name not in models_info:
                print(f"  ⚠️  Model {model_name} not found in fold {fold} metadata")
                total_missing += 1
                continue

            model_info = models_info[model_name]
            steps_per_epoch = model_info.get('steps_per_epoch', 0)
            epochs_used = model_info.get('epochs_used', 0)

            if steps_per_epoch == 0:
                print(f"  ⚠️  Invalid steps_per_epoch for {model_name} in fold {fold}")
                total_missing += 1
                continue

            # Construct history path
            history_path = os.path.join(
                fold_experiment_path, 'models', model_name, 'history.json'
            )

            # Load history
            history_dict = load_history_with_steps(history_path, steps_per_epoch, epochs_used)

            if history_dict is not None:
                histories[model_name][fold] = history_dict
                total_loaded += 1

                # Check for early stopping
                if history_dict['actual_epochs'] < history_dict['expected_epochs']:
                    print(f"  ✓ {model_name}: {history_dict['actual_epochs']} epochs "
                          f"(early stopped from {history_dict['expected_epochs']})")
                else:
                    print(f"  ✓ {model_name}: {history_dict['actual_epochs']} epochs")
            else:
                print(f"  ⚠️  Failed to load history for {model_name}")
                total_missing += 1

    print(f"\n{'='*80}")
    print(f"Summary: {total_loaded} histories loaded, {total_missing} missing")
    print(f"{'='*80}")

    # Validate that total training steps per fold are consistent across models
    print("\n" + "="*80)
    print("VALIDATING TRAINING STEP CONSISTENCY")
    print("="*80)

    for fold in metadata['folds_tested']:
        fold_str = str(fold)
        if fold_str not in metadata['fold_experiments']:
            continue

        models_info = metadata['fold_experiments'][fold_str].get('models', {})
        fold_total_steps = {}

        for model_name in metadata['models_trained']:
            if model_name in models_info:
                model_info = models_info[model_name]
                steps_per_epoch = model_info.get('steps_per_epoch', 0)
                epochs_used = model_info.get('epochs_used', 0)
                total_steps = steps_per_epoch * epochs_used
                fold_total_steps[model_name] = total_steps

        if len(fold_total_steps) > 1:
            # Check consistency (allow 5% threshold)
            steps_values = list(fold_total_steps.values())
            max_steps = max(steps_values)
            min_steps = min(steps_values)

            if max_steps > 0:
                difference_pct = ((max_steps - min_steps) / max_steps) * 100

                if difference_pct > 5.0:
                    print(f"\n⚠️  WARNING: Fold {fold} has inconsistent total steps across models (>{difference_pct:.1f}% difference)")
                    for model_name, steps in sorted(fold_total_steps.items()):
                        print(f"    {model_name}: {steps} steps")
                else:
                    print(f"✓ Fold {fold}: All models have consistent total steps (~{max_steps} steps, {difference_pct:.1f}% variation)")

    return dict(histories)


def get_model_color_map(model_names):
    """
    Create consistent color mapping for models.

    Args:
        model_names: List of model names

    Returns:
        Dictionary mapping model_name -> matplotlib color
    """
    # Define color scheme with Stage1 (lighter) and Stage2 (darker) variants
    color_scheme = {
        'fcn': '#6baed6',           # light blue
        'unet': '#fc8d62',          # light orange
        'imunet': '#66c2a5',        # light green
        'drnet_fcn': '#08519c',     # dark blue
        'drnet_unet': '#e31a1c',    # dark red
        'drnet_imunet': '#238b45'   # dark green
    }

    # Create color map
    color_map = {}
    default_colors = plt.cm.tab10.colors
    default_idx = 0

    for model_name in model_names:
        if model_name in color_scheme:
            color_map[model_name] = color_scheme[model_name]
        else:
            # Use default color cycle for unknown models
            color_map[model_name] = default_colors[default_idx % len(default_colors)]
            default_idx += 1

    return color_map


def plot_model_across_folds(model_name, histories, metadata, output_folder):
    """
    Create multi-panel plot showing one model's training curves across different fold sizes.

    Args:
        model_name: Name of the model
        histories: Nested dictionary of all histories
        metadata: Experiment metadata
        output_folder: Where to save plots
    """
    if model_name not in histories:
        print(f"  ⚠️  No histories found for {model_name}, skipping...")
        return

    model_histories = histories[model_name]
    fold_sizes = sorted(model_histories.keys())

    if len(fold_sizes) == 0:
        print(f"  ⚠️  No fold sizes available for {model_name}, skipping...")
        return

    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Generate dynamic color palette for fold sizes
    n_folds = len(fold_sizes)
    if n_folds <= 10:
        fold_palette = sns.color_palette("tab10", n_folds)
    else:
        fold_palette = sns.color_palette("husl", n_folds)
    fold_colors = {fold_size: fold_palette[idx] for idx, fold_size in enumerate(fold_sizes)}

    # Top subplot: Training Loss
    ax_train = axes[0]
    for fold_size in fold_sizes:
        history = model_histories[fold_size]
        color = fold_colors[fold_size]

        # Plot with markers every 10% of points
        marker_interval = max(1, len(history['steps']) // 10)
        ax_train.plot(history['steps'], history['train_loss'],
                     color=color, linewidth=2, marker='o',
                     markevery=marker_interval, markersize=5,
                     label=f"{fold_size} fold(s)")

    ax_train.set_xlabel('Training Steps', fontweight='bold')
    ax_train.set_ylabel('Training Loss (MSE)', fontweight='bold')
    ax_train.set_title(f'{model_name}: Training Loss vs Steps', fontweight='bold', fontsize=13)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc='best')

    # Bottom subplot: Validation Loss
    ax_val = axes[1]
    for fold_size in fold_sizes:
        history = model_histories[fold_size]
        color = fold_colors[fold_size]

        marker_interval = max(1, len(history['steps']) // 10)
        ax_val.plot(history['steps'], history['val_loss'],
                   color=color, linewidth=2, marker='s',
                   markevery=marker_interval, markersize=5,
                   label=f"{fold_size} fold(s)")

    ax_val.set_xlabel('Training Steps', fontweight='bold')
    ax_val.set_ylabel('Validation Loss (MSE)', fontweight='bold')
    ax_val.set_title(f'{model_name}: Validation Loss vs Steps', fontweight='bold', fontsize=13)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc='best')

    # Overall title
    fig.suptitle(f'Training Dynamics: {model_name} Across Different Training Set Sizes',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save plots
    safe_model_name = model_name.replace('/', '_').replace('\\', '_')
    png_path = os.path.join(output_folder, f'{safe_model_name}_across_folds.png')
    pdf_path = os.path.join(output_folder, f'{safe_model_name}_across_folds.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {png_path}")
    print(f"  ✓ Saved: {pdf_path}")


def plot_fold_comparison(fold_size, histories, metadata, color_map, output_folder):
    """
    Create multi-panel plot comparing all models at a specific fold size.

    Args:
        fold_size: Fold size to compare
        histories: Nested dictionary of all histories
        metadata: Experiment metadata
        color_map: Dictionary mapping model names to colors
        output_folder: Where to save plots
    """
    # Collect models that have data for this fold
    models_with_data = []
    for model_name in metadata['models_trained']:
        if model_name in histories and fold_size in histories[model_name]:
            models_with_data.append(model_name)

    if len(models_with_data) == 0:
        print(f"  ⚠️  No models with data for fold {fold_size}, skipping...")
        return

    # Create figure with 2 rows
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Top subplot: Training Loss Comparison
    ax_train = axes[0]
    for model_name in models_with_data:
        history = histories[model_name][fold_size]
        color = color_map.get(model_name, 'gray')

        marker_interval = max(1, len(history['steps']) // 10)
        ax_train.plot(history['steps'], history['train_loss'],
                     color=color, linewidth=2, marker='o',
                     markevery=marker_interval, markersize=4,
                     label=model_name)

    ax_train.set_xlabel('Training Steps', fontweight='bold')
    ax_train.set_ylabel('Training Loss (MSE)', fontweight='bold')
    ax_train.set_title(f'Training Loss Comparison ({fold_size} fold(s) of data)',
                      fontweight='bold', fontsize=13)
    ax_train.grid(True, alpha=0.3)
    ax_train.legend(loc='best')

    # Bottom subplot: Validation Loss Comparison
    ax_val = axes[1]
    for model_name in models_with_data:
        history = histories[model_name][fold_size]
        color = color_map.get(model_name, 'gray')

        marker_interval = max(1, len(history['steps']) // 10)
        ax_val.plot(history['steps'], history['val_loss'],
                   color=color, linewidth=2, marker='s',
                   markevery=marker_interval, markersize=4,
                   label=model_name)

    ax_val.set_xlabel('Training Steps', fontweight='bold')
    ax_val.set_ylabel('Validation Loss (MSE)', fontweight='bold')
    ax_val.set_title(f'Validation Loss Comparison ({fold_size} fold(s) of data)',
                    fontweight='bold', fontsize=13)
    ax_val.grid(True, alpha=0.3)
    ax_val.legend(loc='best')

    # Overall title
    fig.suptitle(f'Model Comparison with {fold_size} Fold(s) of Training Data',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save plots
    png_path = os.path.join(output_folder, f'fold{fold_size}_model_comparison.png')
    pdf_path = os.path.join(output_folder, f'fold{fold_size}_model_comparison.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {png_path}")
    print(f"  ✓ Saved: {pdf_path}")


def plot_all_models_all_folds_grid(histories, metadata, color_map, output_folder):
    """
    Create comprehensive grid plot showing all models and all folds.

    Args:
        histories: Nested dictionary of all histories
        metadata: Experiment metadata
        color_map: Dictionary mapping model names to colors
        output_folder: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING COMPREHENSIVE GRID PLOT")
    print("="*80)

    models = sorted([m for m in metadata['models_trained'] if m in histories])
    folds = sorted(metadata['folds_tested'])

    if len(models) == 0 or len(folds) == 0:
        print("  ⚠️  No data available for grid plot, skipping...")
        return

    n_models = len(models)
    n_folds = len(folds)

    # Create figure with grid layout
    fig = plt.figure(figsize=(6*n_folds, 4*n_models))
    gs = gridspec.GridSpec(n_models, n_folds, figure=fig, hspace=0.3, wspace=0.3)

    for model_idx, model_name in enumerate(models):
        for fold_idx, fold_size in enumerate(folds):
            ax = fig.add_subplot(gs[model_idx, fold_idx])

            if fold_size in histories[model_name]:
                history = histories[model_name][fold_size]
                color = color_map.get(model_name, 'gray')

                # Plot both train and val loss
                ax.plot(history['steps'], history['train_loss'],
                       color=color, linewidth=1.5, linestyle='-',
                       label='Train', alpha=0.8)
                ax.plot(history['steps'], history['val_loss'],
                       color=color, linewidth=1.5, linestyle='--',
                       label='Val', alpha=0.8)

                ax.set_title(f'{model_name} - {fold_size} fold(s)',
                           fontsize=10, fontweight='bold')
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)

                # Add labels only on edges
                if model_idx == n_models - 1:
                    ax.set_xlabel('Steps', fontsize=9)
                if fold_idx == 0:
                    ax.set_ylabel('Loss (MSE)', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(f'{model_name} - {fold_size} fold(s)',
                           fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

    # Overall title
    fig.suptitle('Training Dynamics: All Models and Training Set Sizes',
                fontsize=16, fontweight='bold', y=0.995)

    # Save plots
    png_path = os.path.join(output_folder, 'all_models_all_folds_grid.png')
    pdf_path = os.path.join(output_folder, 'all_models_all_folds_grid.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")


def plot_convergence_comparison(histories, metadata, color_map, output_folder):
    """
    Create plot showing final validation loss vs training set size for each model.

    Args:
        histories: Nested dictionary of all histories
        metadata: Experiment metadata
        color_map: Dictionary mapping model names to colors
        output_folder: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING CONVERGENCE COMPARISON PLOT")
    print("="*80)

    fig, ax = plt.subplots(figsize=(10, 6))

    models = sorted([m for m in metadata['models_trained'] if m in histories])

    for model_name in models:
        model_histories = histories[model_name]
        fold_sizes = sorted(model_histories.keys())

        if len(fold_sizes) == 0:
            continue

        final_val_losses = [model_histories[fold]['val_loss'][-1] for fold in fold_sizes]

        color = color_map.get(model_name, 'gray')
        ax.plot(fold_sizes, final_val_losses,
               color=color, linewidth=2, marker='o', markersize=8,
               label=model_name)

    ax.set_xlabel('Training Set Size (number of folds)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Final Validation Loss (MSE)', fontweight='bold', fontsize=12)
    ax.set_title('Model Convergence: Final Validation Loss vs Training Set Size',
                fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    ax.set_xticks(sorted(metadata['folds_tested']))

    # Use log scale if losses span multiple orders of magnitude
    # Guard against division by zero
    all_losses = []
    for model_name in models:
        for fold in histories[model_name]:
            all_losses.append(histories[model_name][fold]['val_loss'][-1])

    if len(all_losses) > 0:
        # Filter out zeros and compute ratio
        non_zero_losses = [loss for loss in all_losses if loss > 0]
        if len(non_zero_losses) > 0:
            min_loss = min(non_zero_losses)
            max_loss = max(non_zero_losses)
            loss_ratio = max_loss / min_loss
            if loss_ratio > 10:
                ax.set_yscale('log')
                print("  ℹ️  Using log scale for y-axis (losses span multiple orders of magnitude)")

    plt.tight_layout()

    # Save plots
    png_path = os.path.join(output_folder, 'convergence_comparison.png')
    pdf_path = os.path.join(output_folder, 'convergence_comparison.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")


def plot_learning_rate_schedules(histories, metadata, output_folder):
    """
    Create plot showing learning rate schedules across folds.

    Args:
        histories: Nested dictionary of all histories
        metadata: Experiment metadata
        output_folder: Where to save plots
    """
    print("\n" + "="*80)
    print("CREATING LEARNING RATE SCHEDULES PLOT")
    print("="*80)

    models = sorted([m for m in metadata['models_trained'] if m in histories])
    folds = sorted(metadata['folds_tested'])

    if len(models) == 0 or len(folds) == 0:
        print("  ⚠️  No data available for LR schedules plot, skipping...")
        return

    n_models = len(models)
    n_folds = len(folds)

    # Create figure with grid layout
    fig = plt.figure(figsize=(6*n_folds, 3.5*n_models))
    gs = gridspec.GridSpec(n_models, n_folds, figure=fig, hspace=0.3, wspace=0.3)

    for model_idx, model_name in enumerate(models):
        for fold_idx, fold_size in enumerate(folds):
            ax = fig.add_subplot(gs[model_idx, fold_idx])

            if fold_size in histories[model_name]:
                history = histories[model_name][fold_size]

                ax.plot(history['steps'], history['lr'],
                       color='#1f77b4', linewidth=2)

                ax.set_title(f'{model_name} - {fold_size} fold(s)',
                           fontsize=10, fontweight='bold')
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)

                # Add labels only on edges
                if model_idx == n_models - 1:
                    ax.set_xlabel('Steps', fontsize=9)
                if fold_idx == 0:
                    ax.set_ylabel('Learning Rate', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_title(f'{model_name} - {fold_size} fold(s)',
                           fontsize=10, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

    # Overall title
    fig.suptitle('Learning Rate Schedules Across Training Set Sizes',
                fontsize=16, fontweight='bold', y=0.995)

    # Save plot
    png_path = os.path.join(output_folder, 'learning_rate_schedules.png')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {png_path}")


def create_summary_table(histories, metadata, output_folder):
    """
    Create text summary table of training statistics.

    Args:
        histories: Nested dictionary of all histories
        metadata: Experiment metadata
        output_folder: Where to save table
    """
    print("\n" + "="*80)
    print("CREATING SUMMARY TABLE")
    print("="*80)

    # Collect statistics
    summary_data = []

    for model_name in sorted(histories.keys()):
        for fold_size in sorted(histories[model_name].keys()):
            history = histories[model_name][fold_size]

            initial_train_loss = history['train_loss'][0]
            final_train_loss = history['train_loss'][-1]
            initial_val_loss = history['val_loss'][0]
            final_val_loss = history['val_loss'][-1]
            min_val_loss = np.min(history['val_loss'])

            # Calculate loss reduction percentage (guard against division by zero)
            if initial_val_loss == 0:
                val_loss_reduction_pct = 0.0
                val_loss_reduction_str = 'N/A'
            else:
                val_loss_reduction_pct = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
                val_loss_reduction_str = f'{val_loss_reduction_pct:.2f}'

            # Calculate convergence speed: steps to reach 90% of final loss
            threshold = 0.9 * final_val_loss
            steps_to_90pct = None
            for idx, val_loss in enumerate(history['val_loss']):
                if val_loss <= threshold:
                    steps_to_90pct = history['steps'][idx]
                    break

            steps_to_90pct_str = str(steps_to_90pct) if steps_to_90pct is not None else 'N/A'

            summary_data.append({
                'Model': model_name,
                'Fold_Size': fold_size,
                'Initial_Train_Loss': f'{initial_train_loss:.6f}',
                'Final_Train_Loss': f'{final_train_loss:.6f}',
                'Initial_Val_Loss': f'{initial_val_loss:.6f}',
                'Final_Val_Loss': f'{final_val_loss:.6f}',
                'Min_Val_Loss': f'{min_val_loss:.6f}',
                'Val_Loss_Reduction_%': val_loss_reduction_str,
                'Steps_to_90pct_Final': steps_to_90pct_str,
                'Epochs_Trained': history['actual_epochs'],
                'Total_Steps': history['steps'][-1]
            })

    # Create CSV
    import csv
    csv_path = os.path.join(output_folder, 'training_summary.csv')

    if len(summary_data) > 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)

        print(f"✓ Saved: {csv_path}")

        # Print formatted table
        print("\n" + "-"*80)
        print("TRAINING SUMMARY")
        print("-"*80)
        print(f"{'Model':<20} {'Fold':<6} {'Final Val Loss':<15} {'Min Val Loss':<15} "
              f"{'Reduction %':<12} {'90% Steps':<12} {'Epochs':<8} {'Steps':<10}")
        print("-"*80)

        for row in summary_data:
            print(f"{row['Model']:<20} {row['Fold_Size']:<6} {row['Final_Val_Loss']:<15} "
                  f"{row['Min_Val_Loss']:<15} {row['Val_Loss_Reduction_%']:<12} "
                  f"{row['Steps_to_90pct_Final']:<12} "
                  f"{row['Epochs_Trained']:<8} {row['Total_Steps']:<10}")

        print("-"*80)
    else:
        print("  ⚠️  No data available for summary table")


def generate_all_plots(experiment_folder, output_folder):
    """
    Orchestrate generation of all plots.

    Args:
        experiment_folder: Path to training_data_size_experiment folder
        output_folder: Where to save plots
    """
    # Load metadata
    metadata = load_experiment_metadata(experiment_folder)

    # Load all histories
    histories = load_all_histories(metadata)

    if len(histories) == 0:
        print("\n❌ ERROR: No histories loaded. Cannot generate plots.")
        return

    # Create color map
    color_map = get_model_color_map(metadata['models_trained'])

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n✓ Output folder created: {output_folder}")

    # Generate per-model plots
    print("\n" + "="*80)
    print("GENERATING PER-MODEL PLOTS")
    print("="*80)

    for model_name in metadata['models_trained']:
        if model_name in histories:
            print(f"\nPlotting {model_name}...")
            try:
                plot_model_across_folds(model_name, histories, metadata, output_folder)
            except Exception as e:
                print(f"  ❌ Error plotting {model_name}: {e}")

    # Generate per-fold plots
    print("\n" + "="*80)
    print("GENERATING PER-FOLD PLOTS")
    print("="*80)

    for fold_size in metadata['folds_tested']:
        print(f"\nPlotting fold {fold_size}...")
        try:
            plot_fold_comparison(fold_size, histories, metadata, color_map, output_folder)
        except Exception as e:
            print(f"  ❌ Error plotting fold {fold_size}: {e}")

    # Generate comprehensive plots
    try:
        plot_all_models_all_folds_grid(histories, metadata, color_map, output_folder)
    except Exception as e:
        print(f"\n❌ Error creating grid plot: {e}")

    try:
        plot_convergence_comparison(histories, metadata, color_map, output_folder)
    except Exception as e:
        print(f"\n❌ Error creating convergence plot: {e}")

    try:
        plot_learning_rate_schedules(histories, metadata, output_folder)
    except Exception as e:
        print(f"\n❌ Error creating LR schedules plot: {e}")

    # Generate summary table
    try:
        create_summary_table(histories, metadata, output_folder)
    except Exception as e:
        print(f"\n❌ Error creating summary table: {e}")

    # Print completion message
    print("\n" + "="*80)
    print("PLOT GENERATION COMPLETE!")
    print("="*80)
    print(f"All plots saved to: {output_folder}")
    print("\nGenerated files:")

    # List all generated files
    if os.path.exists(output_folder):
        files = sorted(os.listdir(output_folder))
        for file in files:
            print(f"  - {file}")


def main():
    """Command-line interface and entry point."""
    parser = argparse.ArgumentParser(
        description='Plot training data size experiment loss curves',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python plot_training_data_size_losses.py \\
    --experiment-folder mycode/denoising/output/training_data_size_experiment
        """
    )

    parser.add_argument(
        '--experiment-folder',
        type=str,
        required=True,
        help='Path to training_data_size_experiment folder'
    )

    parser.add_argument(
        '--output-folder',
        type=str,
        default=None,
        help='Override output location (default: {experiment_folder}/results/loss_plots)'
    )

    args = parser.parse_args()

    # Validate experiment folder exists
    if not os.path.exists(args.experiment_folder):
        print(f"❌ ERROR: Experiment folder not found: {args.experiment_folder}")
        sys.exit(1)

    # Validate metadata file exists
    metadata_path = os.path.join(args.experiment_folder, 'experiment_metadata.json')
    if not os.path.exists(metadata_path):
        print(f"❌ ERROR: Metadata file not found: {metadata_path}")
        print("Please ensure the experiment has completed successfully.")
        sys.exit(1)

    # Set default output folder if not specified
    if args.output_folder:
        output_folder = args.output_folder
    else:
        output_folder = os.path.join(args.experiment_folder, 'results', 'loss_plots')

    # Print header
    print("\n" + "="*80)
    print("TRAINING DATA SIZE EXPERIMENT - LOSS CURVE PLOTTING")
    print("="*80)
    print(f"Experiment folder: {args.experiment_folder}")
    print(f"Output folder: {output_folder}")

    try:
        # Generate all plots
        generate_all_plots(args.experiment_folder, output_folder)

        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"All plots have been saved to: {output_folder}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Plot generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
