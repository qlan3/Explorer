import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.experiment import Experiment
from utils.plot import *

def main(argv):
  # python main.py --config_file ./configs/LunarLander.json --config_idx 1
  # python main.py --config_file ./configs/Catcher.json --config_idx 1
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--log_dir', type=str, default='./logs/', help='log directory')
  parser.add_argument('--image_dir', type=str, default='./images/', help='image directory')
  parser.add_argument('--config_file', type=str, default='./configs/dqn_for_CartPole.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
  if not os.path.exists(args.image_dir): os.makedirs(args.image_dir)

  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_from_idx(args.config_idx)
  setattr(cfg, 'log_dir', args.log_dir)
  setattr(cfg, 'image_dir', args.image_dir)

  exp = Experiment(cfg)
  exp.run()

  # Plot results for one agent
  # y_label = 'Rolling Return'
  y_label = 'Return'
  plot_agent_results(exp.train_log_path, exp.image_path, exp.exp_name, y_label, show=False)
  # Show test results for one agent
  show_test_result(exp.test_log_path, y_label)

if __name__=='__main__':
  main(sys.argv)