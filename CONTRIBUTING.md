# Contributing to FPINN-Heat-Equation

Thank you for your interest in contributing! 🎉

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### Suggesting Features

We welcome feature suggestions! Please open an issue with:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes:**
   - Follow PEP 8 style guide
   - Add docstrings to functions
   - Include type hints where possible
   - Add tests if applicable

4. **Test your changes:**
   ```bash
   python src/utils.py
   python src/model.py
   python src/physics.py
   ```

5. **Commit your changes:**
   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork:**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**

## Code Style

- Follow PEP 8 conventions
- Use meaningful variable names
- Add comments for complex logic
- Keep functions focused and small
- Write clear docstrings

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/FPINN-Heat-Equation.git
cd FPINN-Heat-Equation

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/  # if tests are added
```

## Areas for Contribution

- 📊 Additional visualization functions
- 🧪 Unit tests and integration tests
- 📚 Documentation improvements
- 🚀 Performance optimizations
- 🔧 New PDE types
- 🎨 UI/Web interface
- 📦 Docker containerization

## Questions?

Feel free to open an issue for any questions!
