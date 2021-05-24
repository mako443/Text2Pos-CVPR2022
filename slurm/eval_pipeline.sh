#!/bin/bash
#SBATCH --job-name="Eval pipeline"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:11G
#SBATCH --mem=32G
#SBATCH --time=2:29:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source /usr/stud/kolmet/venv/pyg/bin/activate
module load cuda/10.2
srun python3 -m evaluation.pipeline "$@"
