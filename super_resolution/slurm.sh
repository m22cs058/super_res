#!/bin/bash
#
#SBATCH --job-name=dl
#SBATCH --partition=gpu2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#
module load anaconda/3
source activate playground
python train.py