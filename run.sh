#!/bin/bash
#SBATCH --account owner-gpu-guest
#SBATCH --partition notchpeak-gpu-guest
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="p40|a40|a100|a6000|3090"
#SBATCH --time=0:45:00
#SBATCH --mem=30GB
#SBATCH -o proj-%j.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate le-proj

export DOWNLOAD_DIR="/scratch/general/vast/$USER/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/general/vast/$USER/huggingface_cache"
export HF_DATASETS_CACHE="/scratch/general/vast/$USER/huggingface_cache"
export TRANSFORMERS_OFFLINE=1 # cache huggingface stuff offline (its api sucks ass)
export HF_DATASETS_OFFLINE=1

OUT_DIR=/scratch/general/vast/$USER/cs6966/proj/models
python main.py -o ${OUT_DIR}
