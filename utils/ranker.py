import os
import json
import pandas as pd

from utils.helper import *

def rank_test_result(logs_path, ranker_file_dir, env, agent, mode='Test', y_label='Average Return'):
  '''
  Sort the test result based on y_label mean and std values
  '''
  results = []
  log_dir_list = os.listdir(logs_path)
  for log_dir in log_dir_list:
    env_, agent_, config_idx, time_str = log_dir.strip().split('-', 3)
    if (env != env_) or (agent != agent_): 
      # If not a legal log direction, continue
      continue
    if not os.path.isdir(os.path.join(logs_path, log_dir)):
      # If not a directory, continue
      continue
    test_file = os.path.join(logs_path, log_dir, f'results-{mode}.csv')
    if not os.path.isfile(test_file):
      # If test file doesn't exist, continue
      continue
    df = pd.read_csv(test_file)
    y_mean = df[y_label].mean()
    y_std = df[y_label].std(ddof=0)
    result_dict = {'Agent': agent,
                   'Env': env, 
                   'Config Index': config_idx, 
                   'Time': time_str,
                   f'{y_label} (mean)': y_mean,
                   f'{y_label} (std)': y_std}
    results.append(result_dict)
  results = pd.DataFrame(results)
  # Sort according to y_label mean and std value
  results = results.sort_values(by=[f'{y_label} (mean)', f'{y_label} (std)'], ascending=[False, False])
  time_str = get_time_str()
  results_filename = f'SortedResults-{env}-{agent}-{mode}-{time_str}.csv'
  results_path = os.path.join(ranker_file_dir, results_filename)
  results.to_csv(results_path, index=False)
  print(f'{env}-{agent} rank test result Done!')
  
  return results_filename

def csv_good_test_result(logs_path, ranker_file_path, config_file_sweeper, csv_file, topk=10):
  '''
  List good test result into a .csv file
  '''
  results = []
  # Open config file used by sweeper and store sweeper keys into a list
  sweeper_keys = []
  with open(config_file_sweeper, 'r') as f:
    config_dicts = json.load(f)
    for key, values in config_dicts.items():
      if len(values) > 1:
        sweeper_keys.append(key)
  # Open ranker file
  df = pd.read_csv(ranker_file_path)
  for i in range(topk):
    # Read top i test result
    result_dict = df.iloc[i].to_dict()
    # Find config for the config index
    log_dir = f"{result_dict['Env']}-{result_dict['Agent']}-{result_dict['Config Index']}-{result_dict['Time']}"
    config_file = os.path.join(logs_path, log_dir, 'config.json')
    with open(config_file, 'r') as f:
      config_dict = json.load(f)
      for key in sweeper_keys:
        result_dict[key] = config_dict[key]
      results.append(result_dict)
  csv_pd = pd.DataFrame(results)
  csv_pd.to_csv(csv_file, index=False)
  print(f'{csv_file} Done!')

if __name__ == "__main__":
  ranker_file_dir = './analysis/'
  for env in ['Catcher']:
    for agent in ['DQN']:
      logs_path = f'./{env}_logs/'
      ranker_filename = rank_test_result(logs_path, ranker_file_dir, env, agent)
      ranker_file_path = ranker_file_dir + ranker_filename
      config_file_sweeper = f'./configs/{env}-{agent}.json'
      csv_file = ranker_file_dir + f'GoodResults-{env}-{agent}.csv'
      csv_good_test_result(logs_path, ranker_file_path, config_file_sweeper, csv_file, topk=48)