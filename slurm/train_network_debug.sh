#!/bin/bash -l

#SBATCH --job-name=distill_regnet
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=70
#SBATCH --account=a144
#SBATCH --output=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.out
#SBATCH --error=/capstor/store/cscs/swissai/a03/patelm/output_slurm/%x_%j.err
#SBATCH --environment=vllm
#SBATCH --container-workdir=/users/patelm/ws/rsl/gfm_distillation
source ~/.bashrc
ulimit -c 0

# Export required paths
export PYTHONPATH=/users/patelm/ws/rsl/dinov2

cd /users/patelm/ws/rsl/gfm_distillation

echo $PWD
echo "Running distillation script with noise on RegNet"

srun python distill_regnet_webdataset_with_noise.py