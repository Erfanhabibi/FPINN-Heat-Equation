"""
physics.py: Physics-Informed Loss Functions

This module implements the physics-informed loss components for the FPINN:
1. PDE Residual Loss: Enforces the heat equation u_t = k * u_xx
2. Boundary Condition Loss: Enforces BC at x=0 and x=1
3. Initial Condition Loss: Enforces IC at t=0
4. Monotonicity Loss: Ensures u_lower <= u_upper
5. Alpha-Consistency Loss: Ensures proper ordering across alpha levels
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import TriangularFuzzyNumber, initial_condition, boundary_condition


# Define the fuzzy thermal conductivity globally
K_FUZZY = TriangularFuzzyNumber(0.05, 0.1, 0.15)


def compute_derivatives(model, x, t, alpha):
    """
    Compute spatial and temporal derivatives using automatic differentiation.

    Args:
        model: FPINN model
        x, t, alpha: Input tensors with requires_grad=True

    Returns:
        dict: Dictionary containing u, u_t, u_x, u_xx for both lower and upper bounds
    """
    # Ensure gradients are enabled
    x.requires_grad_(True)
    t.requires_grad_(True)

    # Forward pass
    u_lower, u_upper = model(x, t, alpha)

    # Compute derivatives for u_lower
    u_lower_t = torch.autograd.grad(
        u_lower, t,
        grad_outputs=torch.ones_like(u_lower),
        create_graph=True,
        retain_graph=True
    )[0]

    u_lower_x = torch.autograd.grad(
        u_lower, x,
        grad_outputs=torch.ones_like(u_lower),
        create_graph=True,
        retain_graph=True
    )[0]

    u_lower_xx = torch.autograd.grad(
        u_lower_x, x,
        grad_outputs=torch.ones_like(u_lower_x),
        create_graph=True,
        retain_graph=True
    )[0]

    # Compute derivatives for u_upper
    u_upper_t = torch.autograd.grad(
        u_upper, t,
        grad_outputs=torch.ones_like(u_upper),
        create_graph=True,
        retain_graph=True
    )[0]

    u_upper_x = torch.autograd.grad(
        u_upper, x,
        grad_outputs=torch.ones_like(u_upper),
        create_graph=True,
        retain_graph=True
    )[0]

    u_upper_xx = torch.autograd.grad(
        u_upper_x, x,
        grad_outputs=torch.ones_like(u_upper_x),
        create_graph=True,
        retain_graph=True
    )[0]

    return {
        'u_lower': u_lower,
        'u_upper': u_upper,
        'u_lower_t': u_lower_t,
        'u_upper_t': u_upper_t,
        'u_lower_x': u_lower_x,
        'u_upper_x': u_upper_x,
        'u_lower_xx': u_lower_xx,
        'u_upper_xx': u_upper_xx
    }


def pde_residual_loss(model, x, t, alpha, k_fuzzy=K_FUZZY):
    """
    Compute the PDE residual loss for the heat equation.

    The heat equation is: u_t = k * u_xx

    For fuzzy k, we have:
    - u_lower should satisfy: u_lower_t = k_lower * u_lower_xx
    - u_upper should satisfy: u_upper_t = k_upper * u_upper_xx

    Args:
        model: FPINN model
        x, t, alpha: Input tensors
        k_fuzzy: TriangularFuzzyNumber for thermal conductivity

    Returns:
        torch.Tensor: PDE residual loss
    """
    # Compute derivatives
    derivs = compute_derivatives(model, x, t, alpha)

    # Get alpha-cut intervals for k
    alpha_np = alpha.detach().cpu().numpy().flatten()
    k_lower_vals = []
    k_upper_vals = []

    for a in alpha_np:
        k_l, k_u = k_fuzzy.alpha_cut(a)
        k_lower_vals.append(k_l)
        k_upper_vals.append(k_u)

    k_lower = torch.tensor(k_lower_vals, dtype=torch.float32).reshape(-1, 1)
    k_upper = torch.tensor(k_upper_vals, dtype=torch.float32).reshape(-1, 1)

    if x.is_cuda:
        k_lower = k_lower.cuda()
        k_upper = k_upper.cuda()

    # PDE residuals
    # For u_lower: residual = u_lower_t - k_lower * u_lower_xx
    residual_lower = derivs['u_lower_t'] - k_lower * derivs['u_lower_xx']

    # For u_upper: residual = u_upper_t - k_upper * u_upper_xx
    residual_upper = derivs['u_upper_t'] - k_upper * derivs['u_upper_xx']

    # Mean squared error
    loss_lower = torch.mean(residual_lower ** 2)
    loss_upper = torch.mean(residual_upper ** 2)

    return loss_lower + loss_upper


def boundary_condition_loss(model, x_bc, t_bc, alpha_bc):
    """
    Compute boundary condition loss.

    Boundary conditions:
    - u(0, t) = 0 for all t (left boundary)
    - u(1, t) = 0 for all t (right boundary)

    Args:
        model: FPINN model
        x_bc, t_bc, alpha_bc: Boundary condition points

    Returns:
        torch.Tensor: BC loss
    """
    u_lower_bc, u_upper_bc = model(x_bc, t_bc, alpha_bc)

    # Target values (homogeneous Dirichlet BC)
    target = torch.zeros_like(u_lower_bc)

    # MSE loss
    loss_lower = torch.mean((u_lower_bc - target) ** 2)
    loss_upper = torch.mean((u_upper_bc - target) ** 2)

    return loss_lower + loss_upper


def initial_condition_loss(model, x_ic, t_ic, alpha_ic):
    """
    Compute initial condition loss.

    Initial condition: u(x, 0) = sin(pi * x)

    Since the IC is deterministic (no fuzzy uncertainty at t=0),
    both u_lower and u_upper should match the IC.

    Args:
        model: FPINN model
        x_ic, t_ic, alpha_ic: Initial condition points (t_ic should be all zeros)

    Returns:
        torch.Tensor: IC loss
    """
    u_lower_ic, u_upper_ic = model(x_ic, t_ic, alpha_ic)

    # Target: u(x, 0) = sin(pi * x)
    target = initial_condition(x_ic)

    # MSE loss
    loss_lower = torch.mean((u_lower_ic - target) ** 2)
    loss_upper = torch.mean((u_upper_ic - target) ** 2)

    return loss_lower + loss_upper


def monotonicity_loss(model, x, t, alpha):
    """
    Enforce monotonicity constraint: u_lower <= u_upper.

    This ensures that the predicted interval is valid.

    Args:
        model: FPINN model
        x, t, alpha: Input tensors

    Returns:
        torch.Tensor: Monotonicity loss (penalizes u_lower > u_upper)
    """
    u_lower, u_upper = model(x, t, alpha)

    # Penalize cases where u_lower > u_upper
    violation = torch.relu(u_lower - u_upper)
    loss = torch.mean(violation ** 2)

    return loss


def alpha_consistency_loss(model, x, t, alpha_pairs):
    """
    Enforce alpha-consistency: If alpha1 < alpha2, then:
    - u_lower(alpha1) <= u_lower(alpha2)
    - u_upper(alpha1) >= u_upper(alpha2)

    This ensures proper nesting of alpha-cuts.

    Args:
        model: FPINN model
        x, t: Spatial and temporal coordinates
        alpha_pairs: Tensor of shape (N, 2) with pairs (alpha1, alpha2) where alpha1 < alpha2

    Returns:
        torch.Tensor: Alpha-consistency loss
    """
    if alpha_pairs.shape[0] == 0:
        return torch.tensor(0.0)

    alpha1 = alpha_pairs[:, 0:1]
    alpha2 = alpha_pairs[:, 1:2]

    # Predictions at alpha1
    u_lower_1, u_upper_1 = model(x, t, alpha1)

    # Predictions at alpha2
    u_lower_2, u_upper_2 = model(x, t, alpha2)

    # Consistency constraints
    # u_lower should be non-decreasing in alpha
    violation_lower = torch.relu(u_lower_1 - u_lower_2)

    # u_upper should be non-increasing in alpha
    violation_upper = torch.relu(u_upper_2 - u_upper_1)

    loss = torch.mean(violation_lower ** 2) + torch.mean(violation_upper ** 2)

    return loss


def total_loss(model, data_collocation, data_bc, data_ic, alpha_pairs=None, weights=None):
    """
    Compute the total loss as a weighted sum of all loss components.

    Args:
        model: FPINN model
        data_collocation: Dictionary with 'x', 't', 'alpha' for PDE residual
        data_bc: Dictionary with 'x', 't', 'alpha' for boundary conditions
        data_ic: Dictionary with 'x', 't', 'alpha' for initial conditions
        alpha_pairs: Pairs of alpha levels for consistency check (optional)
        weights: Dictionary with loss weights (optional)

    Returns:
        tuple: (total_loss, loss_dict) where loss_dict contains individual losses
    """
    # Default weights
    if weights is None:
        weights = {
            'pde': 1.0,
            'bc': 10.0,
            'ic': 10.0,
            'monotonicity': 1.0,
            'alpha_consistency': 0.5
        }

    # Extract data
    x_col = data_collocation['x']
    t_col = data_collocation['t']
    alpha_col = data_collocation['alpha']

    x_bc = data_bc['x']
    t_bc = data_bc['t']
    alpha_bc = data_bc['alpha']

    x_ic = data_ic['x']
    t_ic = data_ic['t']
    alpha_ic = data_ic['alpha']

    # Compute individual losses
    loss_pde = pde_residual_loss(model, x_col, t_col, alpha_col)
    loss_bc = boundary_condition_loss(model, x_bc, t_bc, alpha_bc)
    loss_ic = initial_condition_loss(model, x_ic, t_ic, alpha_ic)
    loss_mono = monotonicity_loss(model, x_col, t_col, alpha_col)

    # Alpha-consistency loss (optional)
    if alpha_pairs is not None and alpha_pairs.shape[0] > 0:
        loss_alpha = alpha_consistency_loss(model, x_col[:len(alpha_pairs)],
                                            t_col[:len(alpha_pairs)], alpha_pairs)
    else:
        loss_alpha = torch.tensor(0.0)

    # Total weighted loss
    total = (weights['pde'] * loss_pde +
             weights['bc'] * loss_bc +
             weights['ic'] * loss_ic +
             weights['monotonicity'] * loss_mono +
             weights['alpha_consistency'] * loss_alpha)

    # Loss dictionary for logging
    loss_dict = {
        'total': total.item(),
        'pde': loss_pde.item(),
        'bc': loss_bc.item(),
        'ic': loss_ic.item(),
        'monotonicity': loss_mono.item(),
        'alpha_consistency': loss_alpha.item() if isinstance(loss_alpha, torch.Tensor) else 0.0
    }

    return total, loss_dict


if __name__ == "__main__":
    from src.model import FPINN
    from src.utils import generate_training_data_lhs, generate_boundary_conditions

    print("=" * 60)
    print("Physics Loss Functions Test")
    print("=" * 60)

    # Create a model
    model = FPINN(hidden_layers=[20, 20], activation='tanh')
    print(f"\nModel: {model}")

    # Generate test data
    data_col = generate_training_data_lhs(n_samples=50, n_alpha_levels=3)
    data_bc = generate_boundary_conditions(n_bc=20, n_alpha_levels=3)
    data_ic = generate_boundary_conditions(n_bc=20, n_alpha_levels=3)

    # Separate IC points (t=0)
    mask_ic = (data_bc['t'] == 0).squeeze()
    data_ic = {
        'x': data_bc['x'][mask_ic],
        't': data_bc['t'][mask_ic],
        'alpha': data_bc['alpha'][mask_ic]
    }

    # Separate BC points (x=0 or x=1)
    mask_bc = torch.logical_or(data_bc['x'] == 0, data_bc['x'] == 1).squeeze()
    data_bc_filtered = {
        'x': data_bc['x'][mask_bc],
        't': data_bc['t'][mask_bc],
        'alpha': data_bc['alpha'][mask_bc]
    }

    print(f"\nData shapes:")
    print(f"  Collocation: {data_col['x'].shape[0]} points")
    print(f"  Boundary: {data_bc_filtered['x'].shape[0]} points")
    print(f"  Initial: {data_ic['x'].shape[0]} points")

    # Test loss computation
    print("\nComputing losses...")
    total_loss_val, loss_dict = total_loss(
        model, data_col, data_bc_filtered, data_ic
    )

    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.6f}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
