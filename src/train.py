"""
train.py: Training Script for FPINN

This script implements the complete training loop for the Fuzzy Physics-Informed
Neural Network, including:
- Data generation
- Model initialization
- Optimization with Adam
- Loss tracking and logging
- Model checkpointing
"""

from src.utils import (
    generate_training_data_lhs,
    generate_boundary_conditions,
    TriangularFuzzyNumber
)
from src.physics import total_loss
from src.model import FPINN
import torch
import torch.optim as optim
import numpy as np
import time
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class FPINNTrainer:
    """
    Trainer class for FPINN models.
    """

    def __init__(self, model, device='cpu'):
        """
        Initialize the trainer.

        Args:
            model: FPINN model
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.history = {
            'total': [],
            'pde': [],
            'bc': [],
            'ic': [],
            'monotonicity': [],
            'alpha_consistency': []
        }

    def prepare_data(self, n_collocation=2000, n_bc=200, n_alpha_levels=11):
        """
        Prepare training data.

        Args:
            n_collocation: Number of collocation points
            n_bc: Number of boundary/initial points
            n_alpha_levels: Number of alpha levels

        Returns:
            dict: Dictionary with prepared data
        """
        print("Generating training data...")

        # Generate collocation points
        data_col = generate_training_data_lhs(
            n_samples=n_collocation,
            n_alpha_levels=n_alpha_levels
        )

        # Generate boundary/initial condition points
        data_bc_all = generate_boundary_conditions(
            n_bc=n_bc,
            n_alpha_levels=n_alpha_levels
        )

        # Separate IC (t=0) and BC (x=0 or x=1)
        mask_ic = (data_bc_all['t'] == 0).squeeze()
        data_ic = {
            'x': data_bc_all['x'][mask_ic].to(self.device),
            't': data_bc_all['t'][mask_ic].to(self.device),
            'alpha': data_bc_all['alpha'][mask_ic].to(self.device)
        }

        mask_bc = torch.logical_or(
            data_bc_all['x'] == 0,
            data_bc_all['x'] == 1
        ).squeeze()
        data_bc = {
            'x': data_bc_all['x'][mask_bc].to(self.device),
            't': data_bc_all['t'][mask_bc].to(self.device),
            'alpha': data_bc_all['alpha'][mask_bc].to(self.device)
        }

        # Move collocation data to device
        data_col = {
            'x': data_col['x'].to(self.device),
            't': data_col['t'].to(self.device),
            'alpha': data_col['alpha'].to(self.device)
        }

        print(f"  Collocation points: {data_col['x'].shape[0]}")
        print(f"  Boundary points: {data_bc['x'].shape[0]}")
        print(f"  Initial points: {data_ic['x'].shape[0]}")

        return {
            'collocation': data_col,
            'boundary': data_bc,
            'initial': data_ic
        }

    def train(self, n_epochs=10000, lr=1e-3, loss_weights=None,
              print_every=1000, save_path='fpinn_model.pth'):
        """
        Train the FPINN model.

        Args:
            n_epochs: Number of training epochs
            lr: Learning rate
            loss_weights: Dictionary of loss weights
            print_every: Print frequency
            save_path: Path to save the trained model

        Returns:
            dict: Training history
        """
        # Prepare data
        data = self.prepare_data()

        # Initialize optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning rate scheduler (optional)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1000
        )

        print(f"\nStarting training for {n_epochs} epochs...")
        print(f"Learning rate: {lr}")
        print(f"Device: {self.device}")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(n_epochs):
            self.model.train()
            optimizer.zero_grad()

            # Compute loss
            loss, loss_dict = total_loss(
                self.model,
                data['collocation'],
                data['boundary'],
                data['initial'],
                weights=loss_weights
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update scheduler
            scheduler.step(loss)

            # Log history
            for key, value in loss_dict.items():
                self.history[key].append(value)

            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{n_epochs} | "
                      f"Loss: {loss_dict['total']:.6f} | "
                      f"PDE: {loss_dict['pde']:.6f} | "
                      f"BC: {loss_dict['bc']:.6f} | "
                      f"IC: {loss_dict['ic']:.6f} | "
                      f"Time: {elapsed:.2f}s")

        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Training completed in {total_time:.2f}s")

        # Save model
        self.save_model(save_path)

        return self.history

    def save_model(self, path='fpinn_model.pth'):
        """
        Save the trained model.

        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_architecture': {
                'hidden_layers': self.model.hidden_layers,
                'activation': self.model.activation_name
            },
            'history': self.history
        }, path)

        print(f"\nModel saved to: {path}")

    @staticmethod
    def load_model(path='fpinn_model.pth', device='cpu'):
        """
        Load a trained model.

        Args:
            path: Path to the saved model
            device: Device to load the model on

        Returns:
            FPINN: Loaded model
        """
        checkpoint = torch.load(path, map_location=device)

        # Create model with saved architecture
        model = FPINN(
            hidden_layers=checkpoint['model_architecture']['hidden_layers'],
            activation=checkpoint['model_architecture']['activation']
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"Model loaded from: {path}")

        return model, checkpoint.get('history', None)


def main():
    """
    Main training function.
    """
    print("=" * 60)
    print("FPINN Training for 1D Heat Equation")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create model
    model = FPINN(hidden_layers=[50, 50, 50], activation='tanh')
    print(f"\n{model}")

    # Create trainer
    trainer = FPINNTrainer(model, device=device)

    # Loss weights
    loss_weights = {
        'pde': 1.0,
        'bc': 10.0,
        'ic': 10.0,
        'monotonicity': 1.0,
        'alpha_consistency': 0.0  # Can be enabled for stricter training
    }

    # Train the model
    history = trainer.train(
        n_epochs=10000,
        lr=1e-3,
        loss_weights=loss_weights,
        print_every=1000,
        save_path='fpinn_model.pth'
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

    # Print final losses
    print("\nFinal losses:")
    for key in ['total', 'pde', 'bc', 'ic', 'monotonicity']:
        if len(history[key]) > 0:
            print(f"  {key}: {history[key][-1]:.8f}")


if __name__ == "__main__":
    main()
