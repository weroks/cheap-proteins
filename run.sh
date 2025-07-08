#!/bin/bash
#SBATCH --job-name=embeddings

#SBATCH --partition=gpu-7d
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G

#SBATCH --output=embeddings_7d_%j.out
#SBATCH --error=embeddings_7d_%j.out

#SBATCH --mail-type=all
#SBATCH --mail-user=w.klos@tu-berlin.de


apptainer run --nv container.sif python extract_embeddings_hf.py --num-proc 4 --batch-size 16