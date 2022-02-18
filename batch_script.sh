#!/bin/bash
#SBATCH --account=pxs
#SBATCH --time=1:00:00
#SBATCH --job-name="cloud_classification"
#SBATCH --nodes=1
#SBATCH --qos=high
#SBATCH --ntasks-per-node=1
#SBATCH --output='./logs/job_output_filename.%j.out'

module load conda

mkdir -p ./logs

srun /home/bbenton/miniconda3/envs/nsrdb/bin/python cloud_classification.py --batch_run -param_id $1
