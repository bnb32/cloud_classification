#!/bin/bash
#SBATCH --account=pxs
#SBATCH --time=1:00:00
#SBATCH --job-name=cloud_classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=brandon.benton@nrel.gov
#SBATCH --mail-type=END,FAIL
#SBATCH --output=cloud_classification.%j.out

srun ./cloud_classification.sh $1
