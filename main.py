import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment

def main(argv):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/catcher.json', help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=1, help='Configuration index')
  parser.add_argument('--slurm_dir', type=str, default='', help='slurm tempory directory')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  
  # Set config dict default value
  cfg.setdefault('network_update_frequency', 1)
  cfg['env'].setdefault('max_episode_steps', -1)
  cfg.setdefault('show_tb', False)
  cfg.setdefault('render', False)
  cfg.setdefault('gradient_clip', -1)
  cfg.setdefault('hidden_act', 'ReLU')
  cfg.setdefault('output_act', 'Linear')
  

  # Set experiment name and log paths
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]
  if len(args.slurm_dir) > 0:  
    cfg['logs_dir'] = f"{args.slurm_dir}/{cfg['exp']}/{cfg['config_idx']}/"
    make_dir(cfg['logs_dir'])
  else:
    cfg['logs_dir'] = f"./logs/{cfg['exp']}/{cfg['config_idx']}/"
  make_dir(f"./logs/{cfg['exp']}/{cfg['config_idx']}/")
  cfg['train_log_path'] = cfg['logs_dir'] + 'result_Train.feather'
  cfg['test_log_path'] = cfg['logs_dir'] + 'result_Test.feather'
  cfg['model_path'] = cfg['logs_dir'] + 'model.pt'
  cfg['cfg_path'] = cfg['logs_dir'] + 'config.json'

  exp = Experiment(cfg)
  exp.run()

if __name__=='__main__':
  main(sys.argv)