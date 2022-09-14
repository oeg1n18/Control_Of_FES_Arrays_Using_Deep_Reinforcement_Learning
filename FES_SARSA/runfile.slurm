#!/bin/bash

#SBATCH --job-name=SARSA
#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --output=run.out
#SBATCH --error=run.err

python SARSA.py