#!/bin/bash
#SBATCH --job-name="K360 data prepare"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:59:00
#SBATCH --mail-type=NONE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out

source /usr/stud/kolmet/venv/torch/bin/activate
module load cuda/10.2
srun python3 -m datapreparation.kitti360.prepare "$@"
