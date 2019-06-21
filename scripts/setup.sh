#!/usr/bin/env bash

# Get node
salloc --time=24:0:0 --cpus-per-task=48 --account=def-afyshe-ab --mem-per-cpu=1G

# Load singularity
module load singularity/2.6

# Pull the image (if not already exists)
singularity pull --name explorer-env.img shub://qlan3/singularity-deffile:explorer

singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run_Catcher_DQN.sh &
singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run_Catcher_DDQN.sh &
singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run_Catcher_MaxminDQN1.sh &
singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run_Catcher_MaxminDQN2.sh &
singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run_Catcher_MaxminDQN3.sh &
singularity exec -B /project explorer-env.img bash Explorer/scripts/parallel_run_Catcher_MaxminDQN4.sh &
# singularity shell -B /project explorer-env.img