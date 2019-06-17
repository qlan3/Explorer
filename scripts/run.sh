for k in {1..9}
do
  python main.py --config_file ./configs/Catcher-DQN.json --config_idx ${k} &
done