export OMP_NUM_THREADS=1
# git rev-parse --short HEAD
parallel --eta --ungroup --jobs 120 python main.py --config_file ./configs/mujoco_rpg.json --config_idx {1} ::: $(seq 1 360)