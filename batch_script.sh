#!/bin/bash
#SBATCH --account=pxs
#SBATCH --time=4:00:00
#SBATCH --job-name="cloud_classification"
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/bbenton/nsrdb/cloud_classification/logs/cloud_classification.%j.out

module load conda

srun /home/bbenton/miniconda3/envs/nsrdb/bin/python cloud_classification.py --batch_run -param_id $1 -samples $2
