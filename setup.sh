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

tmux new -s catcher-dqn1
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-DQN-logs/ --images_dir ./Catcher-DQN-images/ --config_file ./configs/Catcher-DQN.json --config_idx {1} ::: $(seq 1 480) &

tmux new -s catcher-dqn2
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-DQN-logs/ --images_dir ./Catcher-DQN-images/ --config_file ./configs/Catcher-DQN.json --config_idx {1} ::: $(seq 481 960) &

tmux new -s catcher-ddqn1
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-DDQN-logs/ --images_dir ./Catcher-DDQN-images/ --config_file ./configs/Catcher-DDQN.json --config_idx {1} ::: $(seq 1 480) &

tmux new -s catcher-ddqn2
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-DDQN-logs/ --images_dir ./Catcher-DDQN-images/ --config_file ./configs/Catcher-DDQN.json --config_idx {1} ::: $(seq 481 960) &

tmux new -s catcher-maxmin1
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 1 480) &

tmux new -s catcher-maxmin2
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 481 960) &

tmux new -s catcher-maxmin3
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 961 1440) &

tmux new -s catcher-maxmin4
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 1441 1920) &

tmux new -s catcher-maxmin5
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 1921 2400) &

tmux new -s catcher-maxmin6
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 2401 2880) &

tmux new -s catcher-maxmin7
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 2881 3360) &

tmux new -s catcher-maxmin8
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 3361 3840) &

tmux new -s lunar-dqn1
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./LunarLander-DQN-logs/ --images_dir ./LunarLander-DQN-images/ --config_file ./configs/LunarLander-DQN.json --config_idx {1} ::: $(seq 1 480) &

tmux new -s copter-dqn1
singularity exec -B /project ../explorer-env.img parallel --eta --ungroup python main.py --logs_dir ./Pixelcopter-DQN-logs/ --images_dir ./Pixelcopter-DQN-images/ --config_file ./configs/Pixelcopter-DQN.json --config_idx {1} ::: $(seq 1 480) &