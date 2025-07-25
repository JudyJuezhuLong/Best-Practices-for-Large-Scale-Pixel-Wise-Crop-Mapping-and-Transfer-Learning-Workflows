#!/bin/bash
#SBATCH --partition=mrigpu
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --nodelist=compute-1-4
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=8
#SBATCH --time=50:00:00
#SBATCH --output=/mnt/mridata/Anonymouslong/best_practice_pixel/28_31_Y2022_10_folds/sbatch_output/%x-%j-out.out
#SBATCH --error=/mnt/mridata/Anonymouslong/best_practice_pixel/28_31_Y2022_10_folds/sbatch_output/%x-%j-error.err
#SBATCH -x compute-1-5

# Load the required modules
module use /mnt/it_software/easybuild/modules/all
source /home/jlong2/anaconda3/etc/profile.d/conda.sh

conda env list
conda activate /home/jlong2/anaconda3/envs/env3june28_Anonymous
which python
python --version
echo "Using python from $(which python)"
nvidia-smi
cd /mnt/mridata/Anonymouslong/best_practice_pixel/28_31_Y2022_10_folds
python 8_rf.py