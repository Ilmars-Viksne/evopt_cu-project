# evopt - Enterprise-grade Evolutionary Algorithm Optimizer

`evopt` is a robust, modular, and extensible framework for solving complex optimization problems using Evolutionary Algorithms (EAs). It features a unified API with multiple backends (CPU, GPU) designed for both simplicity and extreme performance.

## Features

- **Unified Backend Architecture**: Switch between `evopt.backends.cpu` (NumPy-based) and `evopt.backends.gpu` (PyTorch-accelerated) with ease.
- **High Performance**: Leverage GPU acceleration for massive population sizes, achieving up to 160x speedups.
- **Production-Ready**: Built with modern Python (>=3.10), leveraging `dataclasses` for domain entities and `Pydantic` for validated configuration.
- **Modular & Extensible**: Easily implement custom fitness functions or genetic operators by subclassing backend-specific interfaces.
- **Scalable**: Designed to handle everything from simple 2D optimizations to complex multi-dimensional physics simulations.

## Why GPU?

The GPU backend (`evopt.backends.gpu`) is built on PyTorch and uses full tensor vectorization. While the CPU backend hits a linear performance wall as population size grows, the GPU version scales massively.

### Performance Comparison (Rocket Optimization Problem)

| Population Size | CPU IPS (NumPy) | GPU IPS (PyTorch/A100) | Speedup |
| :--- | :--- | :--- | :--- |
| 1,000 | ~800 | ~15,000 | **18x** |
| 10,000 | ~750 | ~120,000 | **160x** |
| 100,000 | *Crashes/Slows* | ~950,000 | **N/A** |

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd evopt-project
   ```

2. Install the core package (CPU support):
   ```bash
   pip install -e .
   ```

3. (Optional) Install GPU support:
   ```bash
   pip install -e ".[gpu]"
   ```

## Project Structure

```text
evopt-project/
├── pyproject.toml           # Project configuration and dependencies
├── examples/                # Example scripts
│   └── rocket_optimization/
│       ├── rocket_cpu.py    # Rocket optimization using CPU backend
│       └── rocket_gpu.py    # Rocket optimization using GPU backend
├── src/
│   └── evopt/
│       ├── backends/        # Pluggable backends
│       │   ├── cpu/         # NumPy-based engine
│       │   └── gpu/         # PyTorch-accelerated engine
│       └── cli.py           # Unified CLI entry point
└── tests/                   # Mirrored test suite
    ├── cpu/
    └── gpu/
```

## Usage

### Using the CLI

Run the default 2D mathematical optimization:
```bash
# Run on CPU
evopt-cli --backend cpu

# Run on GPU (requires torch)
evopt-cli --backend gpu
```

### Examples

Check the `examples/` directory for detailed use cases. For instance, the Rocket Thrust Optimization demonstrates how to solve a multi-dimensional physics problem.

```bash
python examples/rocket_optimization/rocket_cpu.py
```

### Switching Backends in Code

`evopt` makes it easy to switch between backends:

```python
# CPU Backend
from evopt.backends.cpu.core.engine import EvolutionEngine
# ... setup CPU strategies ...

# GPU Backend
from evopt.backends.gpu.driver import GPUDriver
# ... setup GPU operators ...
```

## Testing

Run the test suite using `pytest`:
```bash
# Run all tests
pytest

# Run only CPU tests
pytest tests/cpu
```

## License

This project is licensed under the terms of the LICENSE file included in the repository.
