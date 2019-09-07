#!/bin/bash
#SBATCH --job-name=minatar_2
#SBATCH --account=def-afyshe-ab
#SBATCH --time=2-00:00:00
#SBATCH --mem-per-cpu=4000M
#SBATCH --output=output/%x/%j.txt
#SBATCH --mail-user=qlan3@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --mail-type=TIME_LIMIT

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
python main.py --config_file ./configs/minatar_2.json --config_idx $SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------