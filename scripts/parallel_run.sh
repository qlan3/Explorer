cd Explorer/
# Run jobs
parallel python main.py --config_file ./configs/Catcher-DQN.json --config_idx {1} ::: $(seq 1 2)