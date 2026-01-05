"""
utils.py: Fuzzy Number Utilities

This module provides:
- Triangular Fuzzy Number (TFN) representation
- Alpha-cut computation for fuzzy parameters
- Latin Hypercube Sampling (LHS) for training data generation
"""

import numpy as np
import torch
from scipy.stats import qmc


class TriangularFuzzyNumber:
    """
    Triangular Fuzzy Number (TFN) representation.

    A TFN is defined by three parameters (a, b, c):
    - a: lower bound (left endpoint)
    - b: modal value (peak)
    - c: upper bound (right endpoint)

    For thermal conductivity: k_tilde = (0.05, 0.1, 0.15)
    """

    def __init__(self, a, b, c):
        """
        Initialize a Triangular Fuzzy Number.

        Args:
            a (float): Lower bound
            b (float): Modal value
            c (float): Upper bound
        """
        if not (a <= b <= c):
            raise ValueError(
                f"Invalid TFN: must have a <= b <= c, got ({a}, {b}, {c})")

        self.a = a  # Lower bound
        self.b = b  # Modal value
        self.c = c  # Upper bound

    def alpha_cut(self, alpha):
        """
        Compute the alpha-cut interval [k_lower, k_upper] for a given alpha level.

        For a TFN (a, b, c), the alpha-cut is:
        - k_lower(alpha) = a + alpha * (b - a)
        - k_upper(alpha) = c - alpha * (c - b)

        Args:
            alpha (float or np.ndarray): Membership level in [0, 1]

        Returns:
            tuple: (k_lower, k_upper)
        """
        k_lower = self.a + alpha * (self.b - self.a)
        k_upper = self.c - alpha * (self.c - self.b)
        return k_lower, k_upper

    def membership(self, k):
        """
        Compute the membership function value for a given k.

        Args:
            k (float): Value to evaluate

        Returns:
            float: Membership degree in [0, 1]
        """
        if k <= self.a or k >= self.c:
            return 0.0
        elif self.a < k <= self.b:
            return (k - self.a) / (self.b - self.a)
        else:  # self.b < k < self.c
            return (self.c - k) / (self.c - self.b)

    def __repr__(self):
        return f"TFN({self.a}, {self.b}, {self.c})"


def generate_training_data_lhs(n_samples, n_alpha_levels=10, seed=42):
    """
    Generate training data using Latin Hypercube Sampling (LHS).

    This function creates collocation points for:
    - Spatial domain: x in [0, 1]
    - Temporal domain: t in [0, 1]
    - Alpha levels: alpha in [0, 1]

    Args:
        n_samples (int): Number of samples for spatial-temporal domain
        n_alpha_levels (int): Number of alpha levels to discretize [0, 1]
        seed (int): Random seed for reproducibility

    Returns:
        dict: Dictionary containing:
            - 'x': Spatial coordinates (torch.Tensor)
            - 't': Temporal coordinates (torch.Tensor)
            - 'alpha': Alpha levels (torch.Tensor)
            - 'X_collocation': Combined [x, t, alpha] tensor
    """
    # Latin Hypercube Sampling for (x, t)
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples_xt = sampler.random(n=n_samples)

    # Scale to [0, 1] x [0, 1]
    x = samples_xt[:, 0]
    t = samples_xt[:, 1]

    # Generate alpha levels
    alpha_values = np.linspace(0, 1, n_alpha_levels)

    # Create meshgrid: for each (x, t) point, we have all alpha levels
    x_rep = np.repeat(x, n_alpha_levels)
    t_rep = np.repeat(t, n_alpha_levels)
    alpha_rep = np.tile(alpha_values, n_samples)

    # Stack into a single array
    X_collocation = np.column_stack([x_rep, t_rep, alpha_rep])

    # Convert to torch tensors
    data = {
        'x': torch.tensor(x_rep, dtype=torch.float32).reshape(-1, 1),
        't': torch.tensor(t_rep, dtype=torch.float32).reshape(-1, 1),
        'alpha': torch.tensor(alpha_rep, dtype=torch.float32).reshape(-1, 1),
        'X_collocation': torch.tensor(X_collocation, dtype=torch.float32)
    }

    return data


def generate_boundary_conditions(n_bc=100, n_alpha_levels=10):
    """
    Generate boundary condition points.

    For the 1D heat equation on [0, 1]:
    - Boundary at x=0 and x=1 for all t in [0, 1]
    - Initial condition at t=0 for all x in [0, 1]

    Args:
        n_bc (int): Number of boundary points
        n_alpha_levels (int): Number of alpha levels

    Returns:
        dict: Dictionary with boundary condition data
    """
    # Spatial boundary (x=0 and x=1)
    t_bc = np.linspace(0, 1, n_bc)
    x_left = np.zeros(n_bc)
    x_right = np.ones(n_bc)

    # Initial condition (t=0)
    x_ic = np.linspace(0, 1, n_bc)
    t_ic = np.zeros(n_bc)

    # Combine all boundary points
    x_bc_all = np.concatenate([x_left, x_right, x_ic])
    t_bc_all = np.concatenate([t_bc, t_bc, t_ic])

    # Replicate for all alpha levels
    alpha_values = np.linspace(0, 1, n_alpha_levels)
    x_bc_rep = np.repeat(x_bc_all, n_alpha_levels)
    t_bc_rep = np.repeat(t_bc_all, n_alpha_levels)
    alpha_bc_rep = np.tile(alpha_values, len(x_bc_all))

    bc_data = {
        'x': torch.tensor(x_bc_rep, dtype=torch.float32).reshape(-1, 1),
        't': torch.tensor(t_bc_rep, dtype=torch.float32).reshape(-1, 1),
        'alpha': torch.tensor(alpha_bc_rep, dtype=torch.float32).reshape(-1, 1)
    }

    return bc_data


def initial_condition(x):
    """
    Define the initial temperature distribution u(x, 0).

    Using a sine wave: u(x, 0) = sin(pi * x)

    Args:
        x (np.ndarray or torch.Tensor): Spatial coordinates

    Returns:
        Initial temperature values
    """
    if isinstance(x, torch.Tensor):
        return torch.sin(np.pi * x)
    else:
        return np.sin(np.pi * x)


def boundary_condition(t, side='left'):
    """
    Define boundary conditions u(0, t) and u(1, t).

    Using homogeneous Dirichlet: u(0, t) = u(1, t) = 0

    Args:
        t (np.ndarray or torch.Tensor): Temporal coordinates
        side (str): 'left' for x=0, 'right' for x=1

    Returns:
        Boundary values (zeros)
    """
    if isinstance(t, torch.Tensor):
        return torch.zeros_like(t)
    else:
        return np.zeros_like(t)


def compute_analytical_solution(x, t, k):
    """
    Compute the analytical solution for the 1D heat equation with constant k.

    For u_t = k * u_xx with u(x, 0) = sin(pi*x) and homogeneous BC:
    u(x, t) = sin(pi * x) * exp(-k * pi^2 * t)

    Args:
        x (np.ndarray): Spatial coordinates
        t (np.ndarray): Temporal coordinates
        k (float): Thermal conductivity

    Returns:
        np.ndarray: Analytical solution
    """
    return np.sin(np.pi * x) * np.exp(-k * np.pi**2 * t)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Fuzzy Utilities Test")
    print("=" * 60)

    # Define the fuzzy thermal conductivity
    k_fuzzy = TriangularFuzzyNumber(0.05, 0.1, 0.15)
    print(f"\nFuzzy thermal conductivity: {k_fuzzy}")

    # Test alpha-cuts
    alpha_test = [0.0, 0.5, 1.0]
    print("\nAlpha-cut intervals:")
    for alpha in alpha_test:
        k_lower, k_upper = k_fuzzy.alpha_cut(alpha)
        print(f"  α = {alpha:.1f}: k ∈ [{k_lower:.3f}, {k_upper:.3f}]")

    # Generate training data
    print("\nGenerating training data...")
    data = generate_training_data_lhs(n_samples=100, n_alpha_levels=5)
    print(f"  Shape of X_collocation: {data['X_collocation'].shape}")
    print(f"  Sample points (first 5):")
    print(data['X_collocation'][:5])

    # Generate boundary conditions
    print("\nGenerating boundary conditions...")
    bc_data = generate_boundary_conditions(n_bc=50, n_alpha_levels=5)
    print(f"  Number of BC points: {bc_data['x'].shape[0]}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
