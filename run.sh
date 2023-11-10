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
conda activate le-proj

export DOWNLOAD_DIR="/scratch/general/vast/$USER/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/general/vast/$USER/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/$USER/huggingface_cache"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

OUT_DIR=/scratch/general/vast/$USER/cs6966/proj/models
ntfy $SLURM_JOB_ID python main.py -o ${OUT_DIR}
