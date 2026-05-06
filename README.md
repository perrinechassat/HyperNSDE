# HyperNSDE

HyperNSDE is a generative model for synthetic longitudinal health data based on Neural Stochastic Differential Equations. The model is designed to generate longitudinal trajectories while incorporating their relationships with static covariates.

This repository contains the implementation used for the experiments reported in the associated NeurIPS submission.

## Installation

### Prerequisites

The installation requires [Conda](https://docs.conda.io/en/latest/).

### Setup

Clone the repository and create the Conda environment:

```bash
git clone <anonymous-repository-url>
cd HyperNSDE
git submodule update --init --recursive
bash install_env_conda.sh
conda activate env_synth_longi
```

## Project Structure
```bash
HyperNSDE/
├── src/
│   ├── modules/                  # Model building blocks
│   ├── evaluation/               # Evaluation metrics
│   ├── build_module.py           # Model assembly
│   ├── generative_model.py       # Main generative model class
│   ├── losses.py                 # Loss functions
│   ├── hyperopt.py               # Hyperparameter optimization utilities
│   ├── parser.py                 # Argument parser
│   └── utils.py                  # Utility functions
│
├── data_loader/                  # Data loading and preprocessing
│
├── datasets/
│   ├── Simu_OU/                  # Simulated Ornstein-Uhlenbeck datasets
│   ├── VELOUR/                   # Placeholder for VELOUR dataset
│   └── PPMI/                     # Placeholder for PPMI dataset
│
├── experiments/
│   ├── simulations/              # Simulation experiments and Monte Carlo studies
│   └── real_datasets/            # Experiments on real datasets
│
├── benchmark/
│   ├── rtsgan/                   # RTSGAN baseline
│   ├── multiNODEs/               # MultiNODEs baseline
│   └── DGBFGP/                   # DGBFGP baseline
│
├── visualization_results/        # Visualization and evaluation scripts
│
├── main.py                       # Main training entry point
├── run_optuna_hyperopt.py        # Hyperparameter optimization entry point
├── environment.yml               # Conda environment specification
└── install_env_conda.sh          # Environment installation script
```

## Data

The repository includes simulated data used for the synthetic experiments.

Real-world datasets are not included in this repository due to access restrictions. The corresponding folders are provided only as placeholders to indicate the expected structure.

## Usage
### Train a model

```bash
python main.py
```

### Run hyperparameter optimization

```bash
python run_optuna_hyperopt.py
```

Additional experiment-specific configurations are available in:

```bash
experiments/
```

## Reproducibility

The code is organized to reproduce the simulation and real-data experiments described in the paper. Configuration files and experiment scripts are provided in the experiments/ directory.

## Anonymity

This repository has been prepared for anonymous peer review. Author names, institutional information, and links to non-anonymous repositories have been removed.