#!/bin/bash

ENV_NAME="env_synth_longi"  # Change ce nom si besoin

echo "Creating the conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=3.11.8
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing pip and Cython in the environment..."
conda install -y pip
pip install Cython==3.1.2

echo "Installing Synthcity with pip..."
pip install synthcity==0.2.12

echo "Installing other dependencies with pip..."
pip install git+https://github.com/crispitagorico/sigkernel.git
pip install -r requirements.txt

echo "Fetching external Git repositories..."
git submodule update --init --recursive

echo "Upgrading torch and torchvision..."
pip install --upgrade torch torchvision

echo "The environment '$ENV_NAME' is ready."
echo ""
echo "# To activate this environment, use"
echo "#"
echo "#     \$ conda activate $ENV_NAME"
echo "#"
echo "# To deactivate the active environment, use"
echo "#"
echo "#     \$ conda deactivate"
