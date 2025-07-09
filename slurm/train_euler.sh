#!/bin/bash

#SBATCH --ntasks-per-node 4
#SBATCH --nodes 2
#SBATCH --cpus-per-task 16
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpumem:24G
#SBATCH --job-name=distill_imagenet
#SBATCH --output=/cluster/work/rsl/patelm/result/imagenet_distillation/slurm/%x_%j.out
#SBATCH --error=/cluster/work/rsl/patelm/result/imagenet_distillation/slurm/%x_%j.err


source ~/.bashrc
conda activate dinov2_py310

if [ -z "$1" ]; then
    echo "Error: No config file provided."
    exit 1
fi

echo "Selected config file is $1"

# rsync -aP /cluster/scratch/patelm/imagenet $TMPDIR/

# echo "Dataset is copied"

module load eth_proxy

echo "Cuda visible devices ${CUDA_VISIBLE_DEVICES}"

cd /cluster/home/patelm/ws/rsl/gfm_distillation

srun --gres=gpumem:20G python distill_resnet_imagenet_wds.py --config $1
