#!/usr/bin/env bash

# Load singularity
module load singularity/2.5

# Pull the image (if not already exists)
singularity pull explorer-env.img shub://qlan3/singularity-deffiles:explorer

singularity exec -bind /scratch/mzaheer/classic-control explorer-env.img bash Explorer/scripts/parallel_sun.sh