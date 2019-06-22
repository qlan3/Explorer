cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --logs_dir ./Pixelcopter-DQN-logs/ --images_dir ./Pixelcopter-DQN-images/ --config_file ./configs/Pixelcopter-DQN.json --config_idx {1} ::: $(seq 1 96)