#!/bin/bash
#SBATCH --job-name=lunar
#SBATCH --account=rrg-whitem
#SBATCH --time=0-07:00:00
#SBATCH --mem-per-cpu=2000M
#SBATCH --output=output/%x/%a.txt
#SBATCH --mail-user=qlan3@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --mail-type=TIME_LIMIT

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
python main.py --config_file ./configs/lunar.json --config_idx $SLURM_ARRAY_TASK_ID
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------
# Run on Cedar