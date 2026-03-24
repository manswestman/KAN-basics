# KAN-basics

## Overview

Kolmogorov-Arnold Networks replace the fixed activation functions of traditional MLPs with learnable univariate functions placed on the edges of the network (following the Kolmogorov-Arnold representation theorem). This design often leads to better performance on symbolic or low-data mathematical tasks while providing built-in interpretability through the learned spline functions.

This repository delivers:
- A minimal PyTorch implementation of KAN and MLP.
- Step-by-step Jupyter notebooks that walk through data generation, training, and evaluation.
- Pre-generated datasets (Torch tensors) of varying difficulty.
- A `pyproject.toml` setup for easy local installation as an editable package.

The project is intentionally kept simple and educational for students who want to understand KANs without the complexity of full research codebases.

---

## Features

- **Modular architecture**: Clean `KAN.py` and `MLP.py` implementations in `src/models/`.
- **Step-by-step workflow**: Four self-contained notebooks covering the entire pipeline.
- **Reproducible datasets**: Training, test, “extra”, and “hard” splits stored as `.pt` tensors.
- **Package-friendly**: Installable via `pip install -e .` with `pyproject.toml`.
- **MIT licensed** – free to use, modify, and distribute.

---

## File Structure

```text
KAN-basics/
├── data/                          # Pre-generated Torch tensors
│   ├── x_data.pt                  # Training inputs
│   ├── y_data.pt                  # Training targets
│   ├── x_test_data.pt             # Test inputs
│   ├── y_test_data.pt             # Test targets
│   ├── x_data_extra.pt / y_data_extra.pt # Bigger domain and extrapolation
│   └── x_data_hard.pt / y_data_hard.pt   # Bigger domain and extrapolation
│
├── notebooks/                     # Core reproducible workflows
│   ├── 1_data_generation.ipynb    # Synthetic data creation
│   ├── 1_data_generation_OLD.ipynb (legacy)
│   ├── 2_MLP_training.ipynb       # Baseline MLP training
│   ├── 3_KAN_training.ipynb       # KAN training (main demo)
│   └── 4_eval.ipynb               # Comparative evaluation & visualization
│
├── src/
│   └── models/
│       ├── __init__.py
│       ├── KAN.py                 # Kolmogorov-Arnold Network class
│       └── MLP.py                 # Standard MLP for comparison
│
├── pyproject.toml                 # Package configuration (editable install)
├── .gitignore
├── LICENSE                        # MIT License
└── README.md
```

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manswestman/KAN-basics.git
   cd KAN-basics
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. **Install as an editable package**
   ```bash
   pip install -e .
   ```

4. **(Optional) Install notebook dependencies**
   If you plan to run the notebooks interactively, also install:
   ```bash
   pip install jupyter matplotlib torch torchvision torchaudio
   ```

All core dependencies are declared in `pyproject.toml`.

---

## Quick Start

```bash
# Launch the main demo
jupyter lab notebooks/3_KAN_training.ipynb
```

Follow the notebooks in order (1 → 2 → 3 → 4) for the complete story:
1. Understand how the synthetic dataset is generated.
2. Train the baseline MLP.
3. Train the KAN model.
4. Compare results, visualizations, and performance.

All experiments are fully deterministic and reproducible out of the box.

---

## Reproducibility

- Every notebook is self-contained and uses fixed random seeds.
- Datasets are committed to the repository (no external downloads required).
- Training logs and model checkpoints are saved in `src/models/saved/` for easy inspection.
- The entire pipeline can be reproduced on any machine with PyTorch support.

---

## Roadmap / Future Work

- Add symbolic regression / automatic function discovery from learned splines.
- Support for deeper KAN architectures and hyperparameter sweeps.

Contributions and feature requests are welcome!

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Inspired by the original KAN paper: “KAN: Kolmogorov-Arnold Networks” (Liu et al., 2024).
- Built for educational purposes and as the technical foundation for a newsletter series exploring modern neural architectures.

---

**Happy experimenting!**  
If you use this repository in your work or newsletter, feel let me know — I’d love to hear how you’re using KANs. 🚀
