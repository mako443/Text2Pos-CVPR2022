#!/bin/bash
#SBATCH --job-name="Coarse train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=128G
#SBATCH --time=19:59:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

# source /usr/stud/kolmet/venv/pyg/bin/activate
source /usr/stud/kolmet/venv/pyg-2/bin/activate
module load cuda/10.2
srun python3 -m training.coarse "$@"
