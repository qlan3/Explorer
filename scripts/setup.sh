#!/usr/bin/env bash

# Get node
salloc --time=12:0:0 --cpus-per-task=48 --account=def-afyshe-ab --mem=4000M

# Load singularity
module load singularity/2.6

# Pull the image (if not already exists)
singularity pull --name explorer-env.img shub://qlan3/singularity-deffile:explorer

singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run.sh
# singularity shell -B /project explorer-env.img