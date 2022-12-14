#!/bin/bash
#SBATCH --job-name=package_test
#SBATCH --partition gpuq
#SBATCH --cpus-per-task 40
#SBATCH --mail-user=jroy03@nyit.edu
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

module load anaconda


source activate /data/users/jroy03/conda/envs/test_pkgs
python train.py

