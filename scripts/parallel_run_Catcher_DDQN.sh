cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --logs_dir ./Catcher-DDQN-logs/ --images_dir ./Catcher-DDQN-images/ --config_file ./configs/Catcher-DDQN.json --config_idx {1} ::: $(seq 1 48)