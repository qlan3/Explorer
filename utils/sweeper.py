import os
import sys
import json
import argparse

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

class Sweeper(object):
  '''
  This class generates a Config object and corresponding config dict
  given an index and the config file.
  '''
  def __init__(self, config_file):
    with open(config_file, 'r') as f:
      self.config_dicts = json.load(f)
    self.get_num_combinations_of_dict(self.config_dicts)

  def get_num_combinations_of_dict(self, config_dict):
    '''
    Get # of combinations for configurations in a config dict
    '''
    assert type(config_dict) == dict, 'Config file must be a dict!'
    num_combinations_of_dict = 1
    for key, values in config_dict.items():
      num_combinations_of_list = self.get_num_combinations_of_list(values)
      num_combinations_of_dict *= num_combinations_of_list
    config_dict['num_combinations'] = num_combinations_of_dict

  def get_num_combinations_of_list(self, config_list):
    '''
    Get # of combinations for configurations in a config list
    '''
    assert type(config_list) == list, 'Elements in a config dict must be a list!'
    num_combinations_of_list = 0
    for value in config_list:
      if type(value) == dict:
        if not('num_combinations' in value.keys()):
          self.get_num_combinations_of_dict(value)
        num_combinations_of_list += value['num_combinations']
      else:
        num_combinations_of_list += 1
    return num_combinations_of_list

  def generate_config_for_idx(self, idx):
    '''
    Generate a config dict for the index.
    Index is from 1 to # of conbinations.
    '''
    # Get config dict given the index
    cfg = self.get_dict_value(self.config_dicts, (idx-1) % self.config_dicts['num_combinations'])
    # Set config index
    cfg['config_idx'] = idx
    # Set number of combinations
    cfg['num_combinations'] = self.config_dicts['num_combinations']

    return cfg

  def get_list_value(self, config_list, idx):
    for value in config_list:
      if type(value) == dict:
        if idx + 1 - value['num_combinations'] <= 0:
          return self.get_dict_value(value, idx)
        else:
          idx -= value['num_combinations']
      else:
        if idx == 0:
          return value
        else:
          idx -= 1
  
  def get_dict_value(self, config_dict, idx):
    cfg = dict()
    for key, values in config_dict.items():
      if key == 'num_combinations':
        continue
      num_combinations_of_list = self.get_num_combinations_of_list(values)
      value = self.get_list_value(values, idx % num_combinations_of_list)
      cfg[key] = value
      idx = idx // num_combinations_of_list
    return cfg
  
  def print_config_dict(self, config_dict):
    cfg_json = json.dumps(config_dict, indent=2)
    print(cfg_json, end='\n')


if __name__ == "__main__":
  for agent_config in os.listdir('./configs/'):
    config_file = os.path.join('./configs/', agent_config)
    if not os.path.isfile(config_file): continue
    sweeper = Sweeper(config_file)
    # sweeper.print_config_dict(sweeper.config_dicts)
    # sweeper.print_config_dict(sweeper.generate_config_for_idx(213))
    print(f'Number of total combinations in {agent_config}:', sweeper.config_dicts['num_combinations'])