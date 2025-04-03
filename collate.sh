#!/bin/bash

#SBATCH --job-name=collate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH -x xgpi0
#SBATCH --output=collate.slurmlog
#SBATCH --error=collate.slurmlog

/usr/bin/time python3 collate.py