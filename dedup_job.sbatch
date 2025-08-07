#!/bin/bash
#SBATCH -J test                  # Job name
#SBATCH -N 1                             # Total number of nodes requested
#SBATCH -n 16                            # Total number of cores requested
#SBATCH -D /share/garg/accessgaps2024/fsq_cleaning
#SBATCH --get-user-env # retrieve the users login environment
#SBATCH --mem=80000                        # server memory requested (per node)
#SBATCH -t 24:00:00                       # Time limit (hh:mm:ss)
#SBATCH --partition=garg             # Request partition

#SBATCH --output=logs/fsq_dedup_%j.log
#SBATCH --error=logs/fsq_dedup_%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kl877@cornell.edu

module load python/3.10  # Or your cluster's module system
source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh
conda activate wildfires

python3 run_pipeline.py "$@"