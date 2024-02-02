#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00
#SBATCH --error=slurm-%j.err

module load PyTorch-Lightning/1.7.7-foss-2022a
module load Hydra/1.3.2-GCCcore-12.3.0
module load wandb/0.16.1-GCC-12.3.0
module load buildenv/default-foss-2023a-CUDA-12.1.1
module load SciPy-bundle/2023.07-gfbf-2023a

conda create -n ssl_dd -y
conda activate ssl_dd



pip install .
./src/train.py 

