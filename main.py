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

def main(argv):
  # python main.py --config_idx 1 --config_file ./configs/dqn_for_MountainCar.json
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/dqn_for_CartPole.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg, cfg_dict = sweeper.generate_config_from_idx(args.config_idx)

  print(json.dumps(cfg_dict, indent=2), end='\n')

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)