# Quick Start Guide

Get started with FPINN-Heat-Equation in 5 minutes! ⚡

## 1. Prerequisites

- Python 3.8 or higher
- pip package manager

## 2. Installation

```bash
# Clone the repository
git clone https://github.com/Erfanhabibi/FPINN-Heat-Equation.git
cd FPINN-Heat-Equation

# Install dependencies
pip install -r requirements.txt
```

## 3. Train the Model

```powershell
# Set Python path to project directory
$env:PYTHONPATH = (Get-Location).Path
python src/train.py
```

**Training time:** 2-5 minutes (10,000 epochs)

## 4. View Results

Launch the Jupyter notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook includes:
- ✅ Mathematical theory
- ✅ Model loading
- ✅ Inference examples
- ✅ Beautiful visualizations

## 5. Use the Trained Model

```python
from src.train import FPINNTrainer
import numpy as np

# Load the trained model
model, history = FPINNTrainer.load_model('fpinn_model.pth')

# Make predictions
x = np.array([0.5])
t = np.array([0.3])
alpha = np.array([0.0])  # Maximum uncertainty

u_lower, u_upper = model.predict(x, t, alpha)
print(f"u ∈ [{u_lower[0,0]:.6f}, {u_upper[0,0]:.6f}]")
```

## 6. Create Visualizations

```python
from src.visualize import plot_fuzzy_ribbon
import matplotlib.pyplot as plt

# Plot fuzzy ribbons
fig = plot_fuzzy_ribbon(
    model, 
    t_values=[0.2, 0.5, 0.8],
    alpha_levels=[0.0, 0.5, 1.0]
)
plt.show()
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution:** Set PYTHONPATH before running:
```powershell
$env:PYTHONPATH = (Get-Location).Path
python src/train.py
```

### Issue: Missing dependencies

**Solution:** Install all requirements:
```bash
pip install -r requirements.txt
```

### Issue: CUDA not available

**Solution:** The code works fine on CPU! Training takes 2-5 minutes.

## Next Steps

- 📖 Read the full [README.md](README.md)
- 🎓 Complete the [demo.ipynb](notebooks/demo.ipynb) tutorial
- 🔧 Modify model architecture in [src/model.py](src/model.py)
- 🎨 Create custom visualizations in [src/visualize.py](src/visualize.py)
- 🤝 Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Project Structure

```
FPINN-Heat-Equation/
├── src/
│   ├── utils.py       # Fuzzy math & data generation
│   ├── model.py       # FPINN architecture
│   ├── physics.py     # Loss functions
│   ├── train.py       # Training loop
│   └── visualize.py   # Plotting tools
├── notebooks/
│   └── demo.ipynb     # Tutorial notebook
├── fpinn_model.pth    # Trained model (after training)
├── requirements.txt   # Dependencies
└── README.md          # Full documentation
```

## Support

- 📧 Email: erfan.habibi.ehsani@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Erfanhabibi/FPINN-Heat-Equation/issues)
- 📚 Docs: See README.md and demo.ipynb

---

**Happy coding! 🚀**
