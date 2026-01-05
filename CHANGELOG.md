# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-05

### Added
- Initial release of FPINN-Heat-Equation project
- Implementation of Fuzzy Physics-Informed Neural Network (FPINN)
- Support for Triangular Fuzzy Number (TFN) representation
- α-cut decomposition for fuzzy parameter handling
- Complete training pipeline with Adam optimizer
- Physics-informed loss functions:
  - PDE residual loss
  - Boundary condition loss
  - Initial condition loss
  - Monotonicity constraint
  - α-consistency loss
- Comprehensive visualization tools:
  - Fuzzy ribbons
  - 3D surface plots
  - Heatmaps
  - Training history plots
  - Comparison with analytical solutions
- Jupyter notebook tutorial with detailed explanations
- Professional documentation (README, requirements.txt)
- Batch script for easy training execution
- Latin Hypercube Sampling (LHS) for collocation points

### Features
- 3-input (x, t, α) to 2-output (u_lower, u_upper) architecture
- Automatic differentiation using PyTorch autograd
- Model checkpointing and loading
- Training history tracking
- Modular and extensible code structure

### Documentation
- Complete README with mathematical formulation
- Installation and usage instructions
- Theory background on fuzzy numbers and PINNs
- API reference for all modules
- Contributing guidelines
- MIT License

## [Unreleased]

### Planned
- Unit tests and integration tests
- Multi-dimensional heat equations (2D/3D)
- Other PDE types (wave equation, Burgers' equation)
- GPU acceleration optimizations
- Web-based interactive visualization
- Docker containerization
- CI/CD pipeline setup
