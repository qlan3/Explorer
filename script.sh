#!/usr/bin/env bash

# Get node
salloc --time=24:0:0 --cpus-per-task=48 --account=def-afyshe-ab --mem-per-cpu=512M
salloc --time=24:0:0 --cpus-per-task=48 --account=rrg-whitem --mem-per-cpu=512M
salloc --time=24:0:0 --cpus-per-task=48 --account=def-whitem --mem-per-cpu=512M
salloc --time=01:00:0 --cpus-per-task=1 --account=def-whitem --mem-per-cpu=2000M

# Git clone
git clone https://github.com/qlan3/Explorer.git

# Load modules
module load singularity/3.2
module load python/3.7

# Pull the image (if not already exists)
singularity pull --name explorer-env.img shub://qlan3/singularity-deffile:explorer

# Shell in
singularity shell -B /project ../explorer-env.img

# kill 
killall singularity parallel python

# Tensorboard
tensorboard --logdir=./logs/ --host localhost

# Run
python main.py --config_file ./configs/minatar_2.json --config_idx 1
parallel --eta --ungroup python main.py --config_file ./configs/minatar_2.json --config_idx {1} ::: $(seq 1 4)

# Speed
MinAtar: Beluga (60 steps/s) > Graham (35 steps/s) = Cedar (35 steps/s) > Mp2 (10 steps/s)

# zip
tar -cvf logs.tar *.jpg
# unzip
tar -xvf logs.tar
# check without unzip
tar -tf logs.tar