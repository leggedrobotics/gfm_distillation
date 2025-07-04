#!/bin/bash
#SBATCH --job-name=distill_regnet
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=your_partition_name

module load eth_proxy

source activate dinov2_py310

cd /home/patelm/gfm/submodules/rl_nav

python distill_regnet_imagenet.py --config configs/distill_regnet_imagenet.yaml
