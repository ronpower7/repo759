#!/usr/bin/env zsh
#SBATCH --job-name=FirstSlurm
#SBATCH -p instruction
#SBATCH -c 2
#SBATCH --output=FirstSlurm-%j.out
#SBATCH -e FirstSlurm-%j.err


cd $SLURM_SUBMIT_DIR
hostname -I

