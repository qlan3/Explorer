import os
import sys
import time
import json
import math
import argparse
import random
import numpy

from utils.sweeper import Sweeper
from utils.experiment import Experiment
from utils.plot import *

def main(argv):
  # python main.py --config_idx 1 --config_file ./configs/dqn_for_MountainCar.json
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/dqn_for_CartPole.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg, cfg_dict = sweeper.generate_config_from_idx(args.config_idx)
  exp = Experiment(cfg)
  # Update seed in config dict
  cfg_dict['seed'] = cfg.seed
  print(json.dumps(cfg_dict, indent=2), end='\n')
  # Run
  exp.run()
  # Plot results for one agent
  plot_agent_results(exp.log_name, exp.image_name, exp.exp_name)

if __name__=='__main__':
  main(sys.argv)