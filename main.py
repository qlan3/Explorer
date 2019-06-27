import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.experiment import Experiment
from utils.helper import make_dir

def main(argv):
  # python main.py --config_file ./configs/Catcher-DQN.json --config_idx 1
  # python main.py --config_file ./configs/LunarLander-DQN.json --config_idx 1
  # python main.py --config_file ./configs/Pixelcopter-DQN.json --config_idx 1
  # python main.py --config_file ./configs/Breakout-DQN.json --config_idx 1
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/Catcher/DQN.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_from_idx(args.config_idx)
  
  logs_dir = f'./logs/{cfg.game}/{cfg.agent}/{cfg.config_idx}/'
  train_log_path = logs_dir + 'result_Train.csv'
  test_log_path = logs_dir + 'result_Test.csv'
  model_path = logs_dir + 'model.pt'
  cfg_path = logs_dir + 'config.json'

  setattr(cfg, 'logs_dir', logs_dir)
  setattr(cfg, 'train_log_path', train_log_path)
  setattr(cfg, 'test_log_path', test_log_path)
  setattr(cfg, 'model_path', model_path)
  setattr(cfg, 'cfg_path', cfg_path)
  make_dir(cfg.logs_dir)

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)