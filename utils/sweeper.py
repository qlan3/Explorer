import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def unfinished_index(exp, file_name='log.txt', runs=1, max_line_length=10000):
  '''
  Find unfinished config indexes based on the existence of time info in the log file
  '''
  # Read config files
  config_file = f'./configs/{exp}.json'
  sweeper = Sweeper(config_file)
  # Read a list of logs
  print(f'[{exp}]: ', end=' ')
  for i in range(runs * sweeper.config_dicts['num_combinations']):
    log_file = f'./logs/{exp}/{i+1}/{file_name}'
    try:
      with open(log_file, 'r') as f:
        # Get last line
        try:
          f.seek(-max_line_length, os.SEEK_END)
        except IOError:
          # either file is too small, or too many lines requested
          f.seek(0)
        last_line = f.readlines()[-1]
        # Get time info in last line
        try:
          t = float(last_line.split(' ')[-2])
        except:
          print(i+1, end=', ')
          continue
    except:
      print(i+1, end=', ')
      continue
  print()


def time_info(exp, file_name='log.txt', runs=1, nbins=10, max_line_length=10000):
  time_list = []
  # Read config file
  config_file = f'./configs/{exp}.json'
  sweeper = Sweeper(config_file)
  # Read a list of logs
  for i in range(runs * sweeper.config_dicts['num_combinations']):
    log_file = f'./logs/{exp}/{i+1}/{file_name}'
    try:
      with open(log_file, 'r') as f:
        # Get last line
        try:
          f.seek(-max_line_length, os.SEEK_END)
        except IOError:
          # either file is too small, or too many lines requested
          f.seek(0)
        last_line = f.readlines()[-1]
        # Get time info in last line
        try:
          t = float(last_line.split(' ')[-2])
          time_list.append(t)
        except:
          print('No time info in file: '+log_file)
          continue
    except:
      continue
  
  time_list = np.array(time_list)
  print(f'{exp} max time: {np.max(time_list):.2f} minutes')
  print(f'{exp} mean time: {np.mean(time_list):.2f} minutes')
  print(f'{exp} min time: {np.min(time_list):.2f} minutes')
  
  # Plot histogram of time distribution
  from utils.helper import make_dir
  make_dir(f'./logs/{exp}/0/')
  num, bins, patches = plt.hist(time_list, nbins)
  plt.xlabel('Time (min)')
  plt.ylabel('Counts in the bin')
  plt.savefig(f'./logs/{exp}/0/time_info.png')
  # plt.show()
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window
  
  
def memory_info(exp, file_name='log.txt', runs=1, nbins=10, max_line_length=10000):
  mem_list = []
  # Read config file
  config_file = f'./configs/{exp}.json'
  sweeper = Sweeper(config_file)
  # Read a list of logs
  for i in range(runs * sweeper.config_dicts['num_combinations']):
    log_file = f'./logs/{exp}/{i+1}/{file_name}'
    try:
      with open(log_file, 'r') as f:
        # Get last line
        try:
          f.seek(-max_line_length, os.SEEK_END)
        except IOError:
          # either file is too small, or too many lines requested
          f.seek(0)
        last_second_line = f.readlines()[-2]
        # Get memory info in last line
        try:
          m = float(last_second_line.split(' ')[-2])
          mem_list.append(m)
        except:
          print('No memory info in file: '+log_file)
          continue
    except:
      continue
  
  mem_list = np.array(mem_list)
  print(f'{exp} max memory: {np.max(mem_list):.2f} MB')
  print(f'{exp} mean memory: {np.mean(mem_list):.2f} MB')
  print(f'{exp} min memory: {np.min(mem_list):.2f} MB')
  
  # Plot histogram of time distribution
  from utils.helper import make_dir
  make_dir(f'./logs/{exp}/0/')
  num, bins, patches = plt.hist(mem_list, nbins)
  plt.xlabel('Memory (MB)')
  plt.ylabel('Counts in the bin')
  plt.savefig(f'./logs/{exp}/0/memory_info.png')
  # plt.show()
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window

if __name__ == "__main__":
  for agent_config in os.listdir('./configs/'):
    if not '.json' in agent_config:
      continue
    config_file = os.path.join('./configs/', agent_config)
    sweeper = Sweeper(config_file)
    # sweeper.print_config_dict(sweeper.config_dicts)
    # sweeper.print_config_dict(sweeper.generate_config_for_idx(213))
    print(f'Number of total combinations in {agent_config}:', sweeper.config_dicts['num_combinations'])