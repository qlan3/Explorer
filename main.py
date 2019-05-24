import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.experiment import Experiment
from utils.plot import *

def main(argv):
  # python main.py --config_idx 1 --config_file ./configs/dqn_for_MountainCar.json
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--log_dir', type=str, default='./logs/', help='log directory')
  parser.add_argument('--image_dir', type=str, default='./images/', help='image directory')
  parser.add_argument('--model_dir', type=str, default='./models/', help='model directory')
  parser.add_argument('--config_file', type=str, default='./configs/dqn_for_CartPole.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
  if not os.path.exists(args.image_dir): os.makedirs(args.image_dir)
  if not os.path.exists(args.model_dir): os.makedirs(args.model_dir)

  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_from_idx(args.config_idx)
  setattr(cfg, 'log_dir', args.log_dir)
  setattr(cfg, 'image_dir', args.image_dir)
  setattr(cfg, 'model_dir', args.model_dir)

  exp = Experiment(cfg)
  exp.run()
  # Plot results for one agent
  plot_agent_results(exp.log_path, exp.image_path, exp.exp_name)

if __name__=='__main__':
  main(sys.argv)