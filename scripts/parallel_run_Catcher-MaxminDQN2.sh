cd Explorer/
# Run jobs
parallel --eta --ungroup python main.py --config_file ./configs/Catcher-MaxminDQN.json --config_idx {1} ::: $(seq 1513 3024)