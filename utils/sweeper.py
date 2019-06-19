import sys
import json
import argparse
from utils.config import Config

class Sweeper(object):
  '''
  This class generates a Config object and corresponding config dict
  given an index and the config file.
  '''
  def __init__(self, config_file):
    with open(config_file, 'r') as f:
      self.config_dicts = json.load(f)
    self.total_combinations = 1
    self.set_total_combinations()
    print('Total combinations:', self.total_combinations)

  def set_total_combinations(self):
    '''
    Calculate total combinations of configurations
    '''
    self.total_combinations = 1
    for key, values in self.config_dicts.items():
      self.total_combinations *= len(values)

  def generate_config_from_idx(self, idx):
    '''
    Generate the Config object and config dict given the index.
    Index is from 1 to # of total conbinations.
    '''
    cfg = Config()
    # Set config index
    setattr(cfg, 'config_idx', idx)
    idx = (idx-1) % self.total_combinations
    
    for key, values in self.config_dicts.items():
      value_len = len(values)
      value = values[idx % value_len]
      if key in ['lr'] and type(value) == str:
        value = eval(value)
      setattr(cfg, key, value)
      idx = idx // value_len
    
    return cfg