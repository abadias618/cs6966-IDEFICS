#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0:10:00
#SBATCH --mem=65GB
#SBATCH -o proj-%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate locexp

OUT_DIR=/scratch/general/vast/$USER/cs6966/proj/models
python main.py -o ${OUT_DIR}