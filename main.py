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
  # python main.py --config_file ./configs/minatar_2.json --config_idx 1
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/atari_ram1.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  
  # Set config dict default value
  cfg.setdefault('history_length', 4)
  cfg.setdefault('epsilon_decay', 0.999)
  cfg.setdefault('sgd_update_frequency', 1)
  cfg['env'].setdefault('max_episode_steps', -1)
  cfg.setdefault('show_tb', False)
  cfg.setdefault('render', False)
  cfg.setdefault('gradient_clip', -1)

  # Set experiment name and log paths
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
  logs_dir = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  train_log_path = logs_dir + 'result_Train.feather'
  test_log_path = logs_dir + 'result_Test.feather'
  model_path = logs_dir + 'model.pt'
  cfg_path = logs_dir + 'config.json'
  cfg['logs_dir'] = logs_dir
  cfg['train_log_path'] = train_log_path
  cfg['test_log_path'] = test_log_path
  cfg['model_path'] = model_path
  cfg['cfg_path'] = cfg_path

  make_dir(cfg['logs_dir'])

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)