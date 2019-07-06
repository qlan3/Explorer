#!/usr/bin/env bash

# Get node
salloc --time=24:0:0 --cpus-per-task=48 --account=def-afyshe-ab --mem-per-cpu=1G

# Load singularity
module load singularity/2.6

# Pull the image (if not already exists)
singularity pull --name explorer-env.img shub://qlan3/singularity-deffile:explorer

# Change directary
cd Explorer

# Shell in
singularity shell -B /project ../explorer-env.img

# kill 
killall singularity parallel python

# Run
tmux new -s catcher-dqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Catcher/DQN.json --config_idx {1} ::: $(seq 1 60)

tmux new -s catcher-ddqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Catcher/DDQN.json --config_idx {1} ::: $(seq 1 60)

tmux new -s catcher-maxmin
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Catcher/MaxminDQN.json --config_idx {1} ::: $(seq 1 480)

tmux new -s catcher-averaged
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Catcher/AveragedDQN.json --config_idx {1} ::: $(seq 1 480)

tmux new -s lunar-dqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/LunarLander/DQN.json --config_idx {1} ::: $(seq 1 60)

tmux new -s lunar-ddqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/LunarLander/DDQN.json --config_idx {1} ::: $(seq 1 60)

tmux new -s lunar-maxmin
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/LunarLander/MaxminDQN.json --config_idx {1} ::: $(seq 1 960)

tmux new -s lunar-averaged
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/LunarLander/AveragedDQN.json --config_idx {1} ::: $(seq 1 960)

tmux new -s copter-dqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Pixelcopter/DQN.json --config_idx {1} ::: $(seq 1 180)

tmux new -s copter-ddqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Pixelcopter/DDQN.json --config_idx {1} ::: $(seq 1 180)

tmux new -s copter-maxmin
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Pixelcopter/MaxminDQN.json --config_idx {1} ::: $(seq 1 1440)

tmux new -s copter-averaged
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Pixelcopter/AveragedDQN.json --config_idx {1} ::: $(seq 1 1440)

tmux new -s pong-dqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Pong/DQN.json --config_idx {1} ::: $(seq 1 96)

tmux new -s sea-dqn
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --config_file ./configs/Seaquest/DQN.json --config_idx {1} ::: $(seq 1 96)