# HyperNSDE

Implementation of a generative model for synthetic longitudinal health data based on Neural SDEs, incorporating relationships with static covariates.

## ⚙️ Installation

> **Prerequisite**: [Conda](https://docs.conda.io/en/latest/) must be installed.

```bash
git clone https://github.com/anonymous/HyperNSDE.git
cd HyperNSDE
git submodule update --init --recursive
bash install_env_conda.sh
conda activate env_synth_longi
```

## 📁 Project Structure
HyperNSDE/
│
├── src/                        # Core model implementation
│   ├── modules/                # Model building blocks (encoder, decoder, latent model)
│   ├── evaluation/             # Evaluation metrics (fidelity, etc.)
│   ├── build_module.py         # Model assembly
│   ├── generative_model.py     # Main generative model class
│   ├── losses.py               # Loss functions
│   ├── hyperopt.py             # Hyperparameter optimization utilities
│   ├── parser.py               # Argument parser
│   └── utils.py                # Utility functions
│
├── data_loader/                # Data loading and preprocessing
│
├── datasets/                   # Dataset storage
│   ├── Simu_OU/                # Simulated Ornstein-Uhlenbeck datasets
│   ├── VELOUR/                 # VELOUR real dataset (not present)
│   ├── PPMI/                   # PPMI real dataset (not present)
│
├── experiments/                # Experiment configurations and results
│   ├── simulations/            # Simulation experiments & Monte Carlo studies
│   └── real_datasets/          # Experiments on real datasets
│
├── benchmark/                  # Baseline model implementations
│   ├── rtsgan/                 # RTSGAN baseline
│   ├── multiNODEs/             # MultiNODEs baseline
│   ├── DGBFGP/                 # DGBFGP baseline
│
├── visualization_results/      # Result visualization and evaluation scripts
│
├── main.py                     # Training entry point
├── run_optuna_hyperopt.py      # Hyperparameter search entry point
├── environment.yml             # Conda environment specification
└── install_env_conda.sh        # Installation script