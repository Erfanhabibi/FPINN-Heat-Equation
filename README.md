# FPINN-Heat-Equation

**Fuzzy Physics-Informed Neural Networks for the 1D Heat Equation with Uncertain Thermal Conductivity**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project implements a **Fuzzy Physics-Informed Neural Network (FPINN)** to solve the 1D heat equation with uncertain thermal conductivity. The thermal conductivity is modeled as a **Triangular Fuzzy Number (TFN)**, and the network learns interval solutions using **α-cut decomposition**.

### Mathematical Formulation

**PDE:**
```math
\frac{\partial u}{\partial t} = \tilde{k} \frac{\partial^2 u}{\partial x^2}, \quad x \in [0, 1], \; t \in [0, 1]
```

**Fuzzy Parameter:**
```math
\tilde{k} = (0.05, 0.1, 0.15) \quad \text{(Triangular Fuzzy Number)}
```

**Boundary Conditions:**
```math
u(0, t) = 0, \quad u(1, t) = 0
```

**Initial Condition:**
```math
u(x, 0) = \sin(\pi x)
```

**Network Output:**
```math
u(x, t, \alpha) \in [u_{\text{lower}}(x, t, \alpha), u_{\text{upper}}(x, t, \alpha)]
```

---

## 🎯 Features

✅ **Fuzzy Uncertainty Quantification:** Handles fuzzy parameters using α-cut decomposition  
✅ **Physics-Informed Learning:** Enforces PDE, boundary, and initial conditions  
✅ **Monotonicity Constraints:** Ensures valid interval solutions  
✅ **Comprehensive Visualization:** Fuzzy ribbons, 3D surfaces, heatmaps  
✅ **Production-Ready Code:** Modular, documented, and tested  
✅ **Interactive Tutorial:** Jupyter notebook with detailed explanations  

---

## 📁 Project Structure

```
FPINN-Heat-Equation/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── utils.py              # Fuzzy number logic, α-cuts, LHS sampling
│   ├── model.py              # FPINN architecture (3 inputs → 2 outputs)
│   ├── physics.py            # Physics-informed loss functions
│   ├── train.py              # Training script
│   └── visualize.py          # Plotting and visualization tools
├── notebooks/
│   └── demo.ipynb            # Comprehensive tutorial notebook
├── fpinn_model.pth           # Trained model (generated after training)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── LICENSE                   # MIT License
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Erfanhabibi/FPINN-Heat-Equation.git
cd FPINN-Heat-Equation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; import numpy; import matplotlib; print('✓ All dependencies installed!')"
```

---

## 🔧 Usage

### 1. Train the FPINN Model

Run the training script to train the model and save it as `fpinn_model.pth`:

```bash
python src/train.py
```

**Expected Output:**
```
============================================================
FPINN Training for 1D Heat Equation
============================================================

Using device: cuda  # or cpu

Model: FPINN(
  Hidden Layers: [50, 50, 50]
  Activation: tanh
  Total Parameters: 2,852
)

Generating training data...
  Collocation points: 22000
  Boundary points: 4400
  Initial points: 2200

Starting training for 10000 epochs...
============================================================
Epoch 1/10000 | Loss: 1.234567 | PDE: 0.123456 | BC: 0.012345 | IC: 0.001234 | Time: 0.12s
Epoch 1000/10000 | Loss: 0.001234 | PDE: 0.000123 | BC: 0.000012 | IC: 0.000001 | Time: 12.34s
...
============================================================
Training completed in 123.45s

Model saved to: fpinn_model.pth
```

**Training Parameters:**
- **Epochs:** 10,000 (default)
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Loss weights:** PDE=1.0, BC=10.0, IC=10.0, Monotonicity=1.0

### 2. Run the Jupyter Notebook Tutorial

Launch the comprehensive tutorial notebook:

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook includes:
- **Section 1:** Theory and mathematical background
- **Section 2:** Setup and imports
- **Section 3:** Model loading
- **Section 4:** Inference examples at different times and α-levels
- **Section 5:** Advanced visualizations (fuzzy ribbons, 3D surfaces, heatmaps)
- **Section 6:** Conclusion and next steps

### 3. Use the Trained Model for Inference

```python
import torch
import numpy as np
from src.train import FPINNTrainer

# Load the trained model
model, history = FPINNTrainer.load_model('fpinn_model.pth')

# Define test points
x = np.array([0.25, 0.5, 0.75])
t = np.array([0.3, 0.3, 0.3])
alpha = np.array([0.0, 0.0, 0.0])  # Maximum uncertainty

# Make predictions
u_lower, u_upper = model.predict(x, t, alpha)

print("Predictions:")
for i in range(len(x)):
    print(f"  x={x[i]:.2f}, t={t[i]:.2f}: u ∈ [{u_lower[i,0]:.6f}, {u_upper[i,0]:.6f}]")
```

### 4. Visualize Results

```python
from src.visualize import plot_fuzzy_ribbon

# Plot fuzzy ribbons at different times
fig = plot_fuzzy_ribbon(
    model, 
    t_values=[0.2, 0.5, 0.8],
    alpha_levels=[0.0, 0.5, 1.0]
)
plt.show()
```

---

## 📊 Visualization Gallery

### Fuzzy Ribbons
Shows uncertainty bands at different α-levels and time points:
- **α = 0.0:** Maximum uncertainty (full fuzzy set)
- **α = 0.5:** Medium uncertainty
- **α = 1.0:** Modal value (minimum uncertainty)

### 3D Surface Plots
Visualize `u_lower` and `u_upper` as surfaces over the (x, t) domain.

### Heatmaps
2D contour plots showing temperature distribution evolution.

### Training Convergence
Loss curves for PDE residual, boundary/initial conditions, and constraints.

---

## 🧪 Testing Individual Modules

Each module includes a test section that can be run independently:

```bash
# Test utilities
python src/utils.py

# Test model
python src/model.py

# Test physics losses
python src/physics.py

# Test visualization
python src/visualize.py
```

---

## 📚 Theory Background

### Triangular Fuzzy Numbers (TFN)

A TFN is defined by three parameters `(a, b, c)`:
- **a:** Lower bound (minimum value)
- **b:** Modal value (most likely value)
- **c:** Upper bound (maximum value)

**Membership function:**
```math
\mu(k) = \begin{cases}
\frac{k - a}{b - a} & \text{if } a \leq k \leq b \\
\frac{c - k}{c - b} & \text{if } b < k \leq c \\
0 & \text{otherwise}
\end{cases}
```

### α-Cut Decomposition

For a membership level α ∈ [0, 1], the α-cut is:
```math
[\tilde{k}]_\alpha = [k_{\text{lower}}(\alpha), k_{\text{upper}}(\alpha)]
```

For the TFN `(0.05, 0.1, 0.15)`:
```math
k_{\text{lower}}(\alpha) = 0.05 + 0.05\alpha
```
```math
k_{\text{upper}}(\alpha) = 0.15 - 0.05\alpha
```

### FPINN Architecture

**Input Layer:** `[x, t, α]` (3 neurons)  
**Hidden Layers:** `[50, 50, 50]` with Tanh activation  
**Output Layer:** `[u_lower, u_upper]` (2 neurons)

**Total Parameters:** ~2,850

### Loss Function

```math
\mathcal{L} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{BC}} \mathcal{L}_{\text{BC}} + w_{\text{IC}} \mathcal{L}_{\text{IC}} + w_{\text{mono}} \mathcal{L}_{\text{mono}}
```

**Components:**
1. **PDE Residual:** `u_t - k * u_xx = 0`
2. **Boundary Conditions:** `u(0,t) = u(1,t) = 0`
3. **Initial Condition:** `u(x,0) = sin(πx)`
4. **Monotonicity:** `u_lower ≤ u_upper`

---

## 🎓 Key Results

### Analytical Solution (Deterministic)

For constant thermal conductivity `k`, the analytical solution is:
```math
u(x, t) = \sin(\pi x) \exp(-k \pi^2 t)
```

### FPINN Validation

The FPINN predictions are validated against analytical solutions for `k = 0.05, 0.1, 0.15`. The fuzzy ribbon at `α=0` should contain all three analytical curves.

---

## 🔬 Advanced Features

### 1. Custom Architectures

Modify the model architecture in `src/model.py`:

```python
from src.model import FPINN

# Deep network
model = FPINN(hidden_layers=[100, 100, 100, 100], activation='tanh')

# Wide network
model = FPINN(hidden_layers=[200, 200], activation='relu')
```

### 2. Loss Weight Tuning

Adjust loss weights in `src/train.py`:

```python
loss_weights = {
    'pde': 1.0,          # PDE residual
    'bc': 20.0,          # Boundary conditions (increase for stricter BC)
    'ic': 20.0,          # Initial condition
    'monotonicity': 2.0, # u_lower ≤ u_upper constraint
    'alpha_consistency': 1.0  # α-cut nesting property
}
```

### 3. Different Fuzzy Numbers

Modify the fuzzy parameter in `src/utils.py`:

```python
k_fuzzy = TriangularFuzzyNumber(0.08, 0.12, 0.18)  # Different TFN
```

### 4. Adaptive Learning Rate

Enable learning rate scheduling in `src/train.py` (already implemented):

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1000
)
```

---

## 📖 References

1. **Zadeh, L.A.** (1965). "Fuzzy Sets." *Information and Control*, 8(3), 338-353.

2. **Raissi, M., Perdikaris, P., & Karniadakis, G.E.** (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

3. **Hanss, M.** (2005). *Applied Fuzzy Arithmetic: An Introduction with Engineering Applications*. Springer.

4. **Pang, G., Lu, L., & Karniadakis, G.E.** (2019). "fPINNs: Fractional physics-informed neural networks." *SIAM Journal on Scientific Computing*, 41(4), A2603-A2626.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

**Erfan Habibi**

- GitHub: [@Erfanhabibi](https://github.com/Erfanhabibi)
- Email: erfan.habibi.ehsani@gmail.com

---

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- SciML community for inspiration and resources
- George Em Karniadakis group for pioneering PINN research

---

## 📞 Support

For questions, issues, or suggestions:

- **GitHub Issues:** [Create an issue](https://github.com/Erfanhabibi/FPINN-Heat-Equation/issues)
- **Email:** erfan.habibi.ehsani@gmail.com
- **Documentation:** See the Jupyter notebook [notebooks/demo.ipynb](notebooks/demo.ipynb)

---

## 🗺️ Roadmap

- [ ] Multi-dimensional heat equations (2D/3D)
- [ ] Other fuzzy PDE types (wave equation, Burgers' equation)
- [ ] GPU acceleration and optimization
- [ ] Web-based interactive visualization
- [ ] Uncertainty quantification benchmarks
- [ ] Integration with other UQ frameworks
