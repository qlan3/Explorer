#!/bin/bash
# Ask SLURM to send the USR1 signal 120 seconds before end of the time limit
#SBATCH --signal=B:USR1@120
#SBATCH --output=output/%x/%a.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-type=TIME_LIMIT

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs"
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
# ---------------------------------------------------------------------
cleanup()
{
    echo "Copy log files from temporary directory"
    sour=$SLURM_TMPDIR/$SLURM_JOB_NAME/$SLURM_ARRAY_TASK_ID/
    dest=./logs/$SLURM_JOB_NAME/
    echo "Source directory: $sour"
    echo "Destination directory: $dest"
    cp -r $sour $dest
}
# Call `cleanup` once we receive USR1 or EXIT signal
# trap 'cleanup' USR1 EXIT
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
python main.py --config_file ./configs/${SLURM_JOB_NAME}.json --config_idx $SLURM_ARRAY_TASK_ID
# --slurm_dir $SLURM_TMPDIR
# ---------------------------------------------------------------------
echo "Job finished with exit code $? at: `date`"
# ---------------------------------------------------------------------