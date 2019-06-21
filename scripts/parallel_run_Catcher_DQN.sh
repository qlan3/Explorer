cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --log_dir ./Catcher_logs/ --image_dir ./Catcher_images/ --config_file ./configs/Catcher-DQN.json --config_idx {1} ::: $(seq 1 48)