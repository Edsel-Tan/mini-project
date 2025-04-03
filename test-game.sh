#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64gb
##SBATCH --gpus=h100-47
#SBATCH --time=01:00:00
#SBATCH --output=test.slurmlog
#SBATCH --error=test.slurmlog

source ~/.venv/bin/activate
python test-game.py