cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --config_file ./configs/Catcher-DQN.json --config_idx {1} ::: $(seq 1 336)