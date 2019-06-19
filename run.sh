for k in {1..64}
do
  python main.py --config_file ./configs/LunarLander3.json --config_idx ${k}
done