#!/bin/bash

#SBATCH --job-name=generate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8gb
#SBATCH --partition=i7-13700
#SBATCH --time=00:30:00
#SBATCH --output=result.slurmlog
#SBATCH --error=result.slurmlog

/usr/bin/time python3 generate_data.py > result.out
wc -l result.out