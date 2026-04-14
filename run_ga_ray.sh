#!/bin/bash

#SBATCH --job-name=ray
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --account=cis260030p

module load anaconda3
conda activate tf-gpu

export TF_CPP_MIN_LOG_LEVEL=2

python3 ga_ray.py
