#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --gpus=h100-96
#SBATCH --time=00:30:00
#SBATCH --output=result.slurmlog
#SBATCH --error=result.slurmlog

python3 model.py