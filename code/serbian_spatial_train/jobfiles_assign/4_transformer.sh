#!/bin/bash
#SBATCH --partition=mrigpu
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --nodelist=compute-1-3
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH --time=50:00:00
#SBATCH --output=/mnt/mridata/judylong/best_practice_pixel/serbia_spatial_train/sbatch_output/%x-%j-out.out
#SBATCH --error=/mnt/mridata/judylong/best_practice_pixel/serbia_spatial_train/sbatch_output/%x-%j-error.err
#SBATCH -x compute-1-5

# Load the required modules
module use /mnt/it_software/easybuild/modules/all
source /home/jlong2/anaconda3/etc/profile.d/conda.sh

conda env list
conda activate /home/jlong2/anaconda3/envs/env3june28_judy
which python
python --version
echo "Using python from $(which python)"
nvidia-smi
cd /mnt/mridata/judylong/best_practice_pixel/serbia_spatial_train
python 4_transformer.py