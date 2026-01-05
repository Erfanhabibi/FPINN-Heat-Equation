"""
visualize.py: Visualization Tools for FPINN Results

This module provides modular plotting functions for visualizing:
- Fuzzy Ribbons: Uncertainty bands showing u_lower and u_upper
- Temperature distributions at different alpha levels
- Training history and convergence plots
- Comparative analysis with analytical solutions
"""

from src.utils import TriangularFuzzyNumber, compute_analytical_solution
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def plot_fuzzy_ribbon(model, t_values, x_range=(0, 1), n_points=100,
                      alpha_levels=[0.0, 0.5, 1.0], figsize=(15, 5)):
    """
    Plot fuzzy ribbons showing the uncertainty band at different time points.

    Args:
        model: Trained FPINN model
        t_values: List of time values to plot
        x_range: Tuple (x_min, x_max)
        n_points: Number of spatial points
        alpha_levels: List of alpha levels to visualize
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_times = len(t_values)
    fig, axes = plt.subplots(1, n_times, figsize=figsize)

    if n_times == 1:
        axes = [axes]

    x = np.linspace(x_range[0], x_range[1], n_points)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for idx, t in enumerate(t_values):
        ax = axes[idx]

        for i, alpha in enumerate(alpha_levels):
            t_arr = np.full_like(x, t)
            alpha_arr = np.full_like(x, alpha)

            u_lower, u_upper = model.predict(x, t_arr, alpha_arr)
            u_lower = u_lower.flatten()
            u_upper = u_upper.flatten()

            # Plot the fuzzy ribbon
            ax.fill_between(x, u_lower, u_upper, alpha=0.3, color=colors[i],
                            label=f'α = {alpha:.1f}')
            ax.plot(x, u_lower, color=colors[i], linewidth=1.5, linestyle='--')
            ax.plot(x, u_upper, color=colors[i], linewidth=1.5, linestyle='--')

            # Plot the midpoint
            u_mid = (u_lower + u_upper) / 2
            ax.plot(x, u_mid, color=colors[i],
                    linewidth=2, label=f'Mid α={alpha:.1f}')

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x, t)', fontsize=12)
        ax.set_title(f't = {t:.2f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_alpha_cuts(model, x_value, t_value, n_alpha=50, figsize=(10, 6)):
    """
    Plot how the solution interval changes with alpha at a fixed (x, t) point.

    Args:
        model: Trained FPINN model
        x_value: Spatial coordinate
        t_value: Temporal coordinate
        n_alpha: Number of alpha levels
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    alpha_range = np.linspace(0, 1, n_alpha)
    x_arr = np.full(n_alpha, x_value)
    t_arr = np.full(n_alpha, t_value)

    u_lower, u_upper = model.predict(x_arr, t_arr, alpha_range)
    u_lower = u_lower.flatten()
    u_upper = u_upper.flatten()
    u_mid = (u_lower + u_upper) / 2
    u_width = u_upper - u_lower

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Interval bounds
    ax1.plot(alpha_range, u_lower, 'b-', linewidth=2, label='u_lower')
    ax1.plot(alpha_range, u_upper, 'r-', linewidth=2, label='u_upper')
    ax1.fill_between(alpha_range, u_lower, u_upper, alpha=0.3, color='gray')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('α', fontsize=12)
    ax1.set_ylabel('u', fontsize=12)
    ax1.set_title(f'Solution Interval at (x={x_value:.2f}, t={t_value:.2f})',
                  fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Interval width and midpoint
    ax2_twin = ax2.twinx()

    line1 = ax2.plot(alpha_range, u_mid, 'g-', linewidth=2, label='Midpoint')
    line2 = ax2_twin.plot(alpha_range, u_width, 'm--',
                          linewidth=2, label='Width')

    ax2.set_xlabel('α', fontsize=12)
    ax2.set_ylabel('Midpoint', fontsize=12, color='g')
    ax2_twin.set_ylabel('Width', fontsize=12, color='m')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2_twin.tick_params(axis='y', labelcolor='m')
    ax2.set_title('Interval Characteristics', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, fontsize=10)

    plt.tight_layout()
    return fig


def plot_3d_surface(model, alpha=0.0, n_points=50, figsize=(14, 6)):
    """
    Plot 3D surfaces of u_lower and u_upper as functions of (x, t).

    Args:
        model: Trained FPINN model
        alpha: Alpha level to visualize
        n_points: Grid resolution
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, 1, n_points)
    X, T = np.meshgrid(x, t)

    x_flat = X.flatten()
    t_flat = T.flatten()
    alpha_flat = np.full_like(x_flat, alpha)

    u_lower, u_upper = model.predict(x_flat, t_flat, alpha_flat)

    U_lower = u_lower.reshape(n_points, n_points)
    U_upper = u_upper.reshape(n_points, n_points)

    fig = plt.figure(figsize=figsize)

    # Plot u_lower
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, T, U_lower, cmap='viridis', alpha=0.8,
                             edgecolor='none')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('t', fontsize=11)
    ax1.set_zlabel('u_lower', fontsize=11)
    ax1.set_title(f'Lower Bound (α={alpha:.1f})',
                  fontsize=13, fontweight='bold')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Plot u_upper
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, T, U_upper, cmap='plasma', alpha=0.8,
                             edgecolor='none')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('t', fontsize=11)
    ax2.set_zlabel('u_upper', fontsize=11)
    ax2.set_title(f'Upper Bound (α={alpha:.1f})',
                  fontsize=13, fontweight='bold')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    return fig


def plot_training_history(history, figsize=(12, 8)):
    """
    Plot training loss history.

    Args:
        history: Dictionary with loss history
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    loss_keys = ['total', 'pde', 'bc', 'ic',
                 'monotonicity', 'alpha_consistency']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']

    for idx, (key, color) in enumerate(zip(loss_keys, colors)):
        if key in history and len(history[key]) > 0:
            ax = axes[idx]
            losses = history[key]
            epochs = np.arange(1, len(losses) + 1)

            ax.plot(epochs, losses, color=color, linewidth=2)
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel('Loss', fontsize=11)
            ax.set_title(f'{key.upper()} Loss', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

    plt.tight_layout()
    return fig


def plot_comparison_with_analytical(model, t_values, k_values=[0.05, 0.1, 0.15],
                                    n_points=100, figsize=(15, 5)):
    """
    Compare FPINN predictions with analytical solutions.

    Args:
        model: Trained FPINN model
        t_values: List of time values
        k_values: List of conductivity values [k_lower, k_modal, k_upper]
        n_points: Number of spatial points
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    n_times = len(t_values)
    fig, axes = plt.subplots(1, n_times, figsize=figsize)

    if n_times == 1:
        axes = [axes]

    x = np.linspace(0, 1, n_points)

    for idx, t in enumerate(t_values):
        ax = axes[idx]

        # FPINN prediction at alpha=0 (widest interval)
        t_arr = np.full_like(x, t)
        alpha_arr = np.full_like(x, 0.0)
        u_lower_fpinn, u_upper_fpinn = model.predict(x, t_arr, alpha_arr)
        u_lower_fpinn = u_lower_fpinn.flatten()
        u_upper_fpinn = u_upper_fpinn.flatten()

        # Plot FPINN fuzzy ribbon
        ax.fill_between(x, u_lower_fpinn, u_upper_fpinn, alpha=0.3,
                        color='blue', label='FPINN (α=0)')

        # Plot analytical solutions for different k values
        colors_analytical = ['red', 'green', 'orange']
        labels = ['k=0.05 (min)', 'k=0.10 (modal)', 'k=0.15 (max)']

        for k, color, label in zip(k_values, colors_analytical, labels):
            u_analytical = compute_analytical_solution(x, t, k)
            ax.plot(x, u_analytical, color=color, linestyle='--',
                    linewidth=2, label=label)

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('u(x, t)', fontsize=12)
        ax.set_title(f't = {t:.2f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    return fig


def plot_heatmap(model, alpha=0.0, bound='lower', n_points=100, figsize=(10, 8)):
    """
    Plot a 2D heatmap of the solution in the (x, t) space.

    Args:
        model: Trained FPINN model
        alpha: Alpha level
        bound: 'lower' or 'upper'
        n_points: Grid resolution
        figsize: Figure size

    Returns:
        matplotlib.figure.Figure: The created figure
    """
    x = np.linspace(0, 1, n_points)
    t = np.linspace(0, 1, n_points)
    X, T = np.meshgrid(x, t)

    x_flat = X.flatten()
    t_flat = T.flatten()
    alpha_flat = np.full_like(x_flat, alpha)

    u_lower, u_upper = model.predict(x_flat, t_flat, alpha_flat)

    if bound == 'lower':
        U = u_lower.reshape(n_points, n_points)
        title = f'Lower Bound (α={alpha:.1f})'
        cmap = 'viridis'
    else:
        U = u_upper.reshape(n_points, n_points)
        title = f'Upper Bound (α={alpha:.1f})'
        cmap = 'plasma'

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.contourf(X, T, U, levels=50, cmap=cmap)
    ax.set_xlabel('x', fontsize=13)
    ax.set_ylabel('t', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('u', fontsize=12)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("Visualization Tools Test")
    print("=" * 60)
    print("\nThis module provides plotting functions for FPINN results.")
    print("Use these functions in the Jupyter notebook or training scripts.")
    print("\nAvailable functions:")
    print("  - plot_fuzzy_ribbon(): Uncertainty bands at different times")
    print("  - plot_alpha_cuts(): Solution intervals vs alpha")
    print("  - plot_3d_surface(): 3D visualization of solutions")
    print("  - plot_training_history(): Loss convergence")
    print("  - plot_comparison_with_analytical(): Compare with exact solutions")
    print("  - plot_heatmap(): 2D heatmaps in (x, t) space")
    print("=" * 60)
