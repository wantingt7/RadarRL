#!/usr/bin/env bash

#SBATCH -J test
#SBATCH -p p-RTX2080
#SBATCH --gres=gpu:1


python3 -u run_experiments.py