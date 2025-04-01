#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --gpus=a100-80
#SBATCH --time=00:30:00
#SBATCH -x xgpi0
#SBATCH --output=result.slurmlog
#SBATCH --error=result.slurmlog

source ~/.venv/bin/activate
echo Allocated.
python3 train.py