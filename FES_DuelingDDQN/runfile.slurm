#!/bin/bash

#SBATCH --job-name=train_control_policy
#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=run.out
#SBATCH --error=run.err

python DeulingDDQN.py