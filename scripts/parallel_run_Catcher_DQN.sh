cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --logs_dir ./Catcher-DQN-logs/ --images_dir ./Catcher-DQN-images/ --config_file ./configs/Catcher-DQN.json --config_idx {1} ::: $(seq 1 48)