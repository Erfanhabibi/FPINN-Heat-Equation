"""
model.py: FPINN Architecture

This module defines the Fuzzy Physics-Informed Neural Network (FPINN) for solving
the 1D heat equation with fuzzy thermal conductivity.

Network Structure:
- Input: [x, t, alpha] (3 features)
- Output: [u_lower, u_upper] (2 features representing the interval solution)
"""

import torch
import torch.nn as nn
import numpy as np


class FPINN(nn.Module):
    """
    Fuzzy Physics-Informed Neural Network.

    This network learns the interval solution [u_lower, u_upper] for the 1D heat
    equation with uncertain thermal conductivity represented as a fuzzy number.

    Architecture:
    - Input layer: 3 neurons (x, t, alpha)
    - Hidden layers: Fully connected with activation functions
    - Output layer: 2 neurons (u_lower, u_upper)
    """

    def __init__(self, hidden_layers=[50, 50, 50], activation='tanh'):
        """
        Initialize the FPINN model.

        Args:
            hidden_layers (list): List of integers specifying hidden layer sizes
            activation (str): Activation function ('tanh', 'relu', 'sigmoid')
        """
        super(FPINN, self).__init__()

        self.hidden_layers = hidden_layers
        self.activation_name = activation

        # Define activation function
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        # Build the network architecture
        layers = []
        input_dim = 3  # [x, t, alpha]

        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            input_dim = hidden_dim

        # Output layer (2 outputs: u_lower, u_upper)
        layers.append(nn.Linear(input_dim, 2))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier (Glorot) initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t, alpha):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Spatial coordinates, shape (N, 1)
            t (torch.Tensor): Temporal coordinates, shape (N, 1)
            alpha (torch.Tensor): Alpha levels, shape (N, 1)

        Returns:
            tuple: (u_lower, u_upper) each of shape (N, 1)
        """
        # Concatenate inputs
        inputs = torch.cat([x, t, alpha], dim=1)

        # Forward pass
        outputs = self.network(inputs)

        # Split outputs
        u_lower = outputs[:, 0:1]
        u_upper = outputs[:, 1:2]

        return u_lower, u_upper

    def predict(self, x, t, alpha):
        """
        Make predictions without gradient computation.

        Args:
            x (torch.Tensor or np.ndarray): Spatial coordinates
            t (torch.Tensor or np.ndarray): Temporal coordinates
            alpha (torch.Tensor or np.ndarray): Alpha levels

        Returns:
            tuple: (u_lower, u_upper) as numpy arrays
        """
        self.eval()

        # Convert to tensors if needed
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(t, np.ndarray):
            t = torch.tensor(t, dtype=torch.float32)
        if isinstance(alpha, np.ndarray):
            alpha = torch.tensor(alpha, dtype=torch.float32)

        # Ensure proper shape
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        if t.dim() == 1:
            t = t.reshape(-1, 1)
        if alpha.dim() == 1:
            alpha = alpha.reshape(-1, 1)

        with torch.no_grad():
            u_lower, u_upper = self.forward(x, t, alpha)

        return u_lower.numpy(), u_upper.numpy()

    def count_parameters(self):
        """
        Count the total number of trainable parameters.

        Returns:
            int: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        """String representation of the model."""
        n_params = self.count_parameters()
        return (f"FPINN(\n"
                f"  Hidden Layers: {self.hidden_layers}\n"
                f"  Activation: {self.activation_name}\n"
                f"  Total Parameters: {n_params:,}\n"
                f")")


class FPINNEnsemble:
    """
    Ensemble of FPINN models for improved predictions.

    This can be used for uncertainty quantification and robustness.
    """

    def __init__(self, n_models=5, **model_kwargs):
        """
        Initialize an ensemble of FPINN models.

        Args:
            n_models (int): Number of models in the ensemble
            **model_kwargs: Arguments to pass to each FPINN model
        """
        self.n_models = n_models
        self.models = [FPINN(**model_kwargs) for _ in range(n_models)]

    def predict(self, x, t, alpha):
        """
        Make ensemble predictions by averaging over all models.

        Args:
            x, t, alpha: Input coordinates

        Returns:
            tuple: Mean predictions (u_lower_mean, u_upper_mean)
        """
        predictions_lower = []
        predictions_upper = []

        for model in self.models:
            u_lower, u_upper = model.predict(x, t, alpha)
            predictions_lower.append(u_lower)
            predictions_upper.append(u_upper)

        u_lower_mean = np.mean(predictions_lower, axis=0)
        u_upper_mean = np.mean(predictions_upper, axis=0)

        return u_lower_mean, u_upper_mean


def create_model(architecture='default'):
    """
    Factory function to create FPINN models with predefined architectures.

    Args:
        architecture (str): Architecture type ('default', 'deep', 'wide')

    Returns:
        FPINN: Initialized model
    """
    if architecture == 'default':
        return FPINN(hidden_layers=[50, 50, 50], activation='tanh')
    elif architecture == 'deep':
        return FPINN(hidden_layers=[50, 50, 50, 50, 50], activation='tanh')
    elif architecture == 'wide':
        return FPINN(hidden_layers=[100, 100, 100], activation='tanh')
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    print("=" * 60)
    print("FPINN Model Test")
    print("=" * 60)

    # Create a model
    model = FPINN(hidden_layers=[50, 50, 50], activation='tanh')
    print(f"\n{model}")

    # Test forward pass
    batch_size = 10
    x = torch.rand(batch_size, 1)
    t = torch.rand(batch_size, 1)
    alpha = torch.rand(batch_size, 1)

    print(f"\nInput shapes:")
    print(f"  x: {x.shape}")
    print(f"  t: {t.shape}")
    print(f"  alpha: {alpha.shape}")

    u_lower, u_upper = model(x, t, alpha)

    print(f"\nOutput shapes:")
    print(f"  u_lower: {u_lower.shape}")
    print(f"  u_upper: {u_upper.shape}")

    print(f"\nSample outputs (first 3):")
    for i in range(min(3, batch_size)):
        print(
            f"  Point {i+1}: u ∈ [{u_lower[i].item():.4f}, {u_upper[i].item():.4f}]")

    # Test prediction method
    print("\nTesting prediction method...")
    x_np = np.random.rand(5)
    t_np = np.random.rand(5)
    alpha_np = np.random.rand(5)

    u_lower_pred, u_upper_pred = model.predict(x_np, t_np, alpha_np)
    print(f"  Prediction successful! Shape: {u_lower_pred.shape}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
