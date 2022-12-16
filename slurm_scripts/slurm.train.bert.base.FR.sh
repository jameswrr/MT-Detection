#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition gpuq
#SBATCH --cpus-per-task 40
#SBATCH --mail-user=name@domain.co.uk
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

module load anaconda
mdule load cuda10.1


source activate /data/users/jroy03/conda/envs/CONDA_ENV
python train.py -m bert-base-uncased -d data/splits/FR -c train.config.distil.FR.json

