# Covariates-Aware-LongiGen

This repository contains the implementation of a generative model for synthetic longitudinal health data based on Neural SDEs, incorporating relationships with static covariates.

## 🚀 Clone this repository

Some parts of this project depend on external Git repositories that are included as submodules.
```bash
git clone https://github.com/perrinechassat/Covariates-Aware-LongiGen.git
cd Covariates-Aware-LongiGen
git submodule update --init --recursive
```

## ⚙️ Installation

> **Prerequisite**: Make sure [Conda] is installed on your system.

This project uses a Conda environment. To set it up, run:
```bash
bash install_env_conda.sh
conda activate env_synth_longi
```
<!-- 
The script will:
- Create a Conda environment named env_synth_longi
- Install Cython, synthcity, and other required pip packages
- Initialize external repositories via submodules
- Upgrade torch and torchvision to the latest compatible versions -->

<!-- ## Project structure

project_root/
│
├── data_loader/             # Scripts or files related to data loading and preprocessing
│
├── datasets/                 
│
├── exp_simulated_data/                
│
├── src/                
│
├── main.py                 
├── run_optuna_hyperopt.py      
├── environment.yml          # Conda environment specification file
├── README.md   -->
