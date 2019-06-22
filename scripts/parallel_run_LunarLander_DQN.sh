cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --logs_dir ./LunarLander-DQN-logs/ --images_dir ./LunarLander-DQN-images/ --config_file ./configs/LunarLander-DQN.json --config_idx {1} ::: $(seq 1 50)