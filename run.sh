#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00
#SBATCH --error=slurm-%j.err

ENV_NAME=ssl_dd



module load Miniconda3/23.9.0-0


module load PyTorch-Lightning/1.7.7-foss-2022a


source $(conda info --base)/etc/profile.d/conda.sh
if ! conda env list | grep -q $ENV_NAME; then
  conda create -n ssl_dd -y
fi
conda activate $ENV_NAME



pip install .
chmod +x src/train.py
python ./src/train.py

