#!/bin/bash
#SBATCH -J test
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH --ntasks 1


python3 -u test_dqn.py