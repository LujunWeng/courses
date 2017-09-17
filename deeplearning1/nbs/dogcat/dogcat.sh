#!/bin/sh

#SBATCH --cpus-per-task=16
#SBATCH --mem=30GB

python dogcat_redux.py

