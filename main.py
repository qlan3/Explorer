import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.experiment import Experiment

def main(argv):
  # python main.py --config_file ./configs/Catcher-DQN.json --config_idx 1
  # python main.py --config_file ./configs/LunarLander-DQN.json --config_idx 1
  # python main.py --config_file ./configs/Pixelcopter-DQN.json --config_idx 1
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--logs_dir', type=str, default='./Catcher-logs/', help='log directory')
  parser.add_argument('--images_dir', type=str, default='./Catcher-images/', help='image directory')
  parser.add_argument('--config_file', type=str, default='./configs/Catcher-DQN.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  if not os.path.exists(args.logs_dir): os.makedirs(args.logs_dir, exist_ok=True)
  if not os.path.exists(args.images_dir): os.makedirs(args.images_dir, exist_ok=True)

  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_from_idx(args.config_idx)
  setattr(cfg, 'logs_dir', args.logs_dir)
  setattr(cfg, 'images_dir', args.images_dir)

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)
