#!/bin/bash

#SBATCH --job-name=Qlearning
#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=run.out
#SBATCH --error=run.err

python TabularQLearning.py