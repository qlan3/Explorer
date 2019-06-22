cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --logs_dir ./Catcher-MaxminDQN-logs/ --images_dir ./Catcher-MaxminDQN-images/ --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 1 48)