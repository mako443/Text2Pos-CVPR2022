#!/bin/bash
#SBATCH --job-name="Cell retrieval train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1,VRAM:32G
#SBATCH --mem=32G
#SBATCH --time=2:59:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source /usr/stud/kolmet/venv/pyg/bin/activate
module load cuda/10.2
srun python3 -m training.cell_retrieval "$@"
