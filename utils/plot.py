import os
import json
import pandas as pd
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt


def get_total_combination(game, agent, config_idx=1):
  '''
  Get total combination of configs in ./config/
  '''
  logs_path = f'./{game}-{agent}-logs/'
  log_dir = f'{config_idx}'
  assert os.path.isdir(os.path.join(logs_path, log_dir)), f'No log dir <{log_dir}>!'
  config_path = os.path.join(logs_path, log_dir, 'config.json')
  with open(config_path, 'r') as f:
    config_dict = json.load(f)
  return config_dict['total_combinations']

def get_sweep_keys(game, agent):
  '''
  Open sweep config file and return sweep keys as a list
  '''
  sweep_config_file = f'./configs/{game}-{agent}.json'
  sweep_keys = []
  with open(sweep_config_file, 'r') as f:
    config_dicts = json.load(f)
    for key, values in config_dicts.items():
      if len(values) > 1:
        sweep_keys.append(key)
  
  return sweep_keys

def merge_runs_results(game, agent, config_idx, total_combination, mode):
  '''
  Given game, agent and index, merge the results for multiple runs
  '''
  logs_path = f'./{game}-{agent}-logs/'
  results = None
  while True:
    log_dir = f'{config_idx}'
    result_path = os.path.join(logs_path, log_dir, f'result-{mode}.csv')
    # If result file doesn't exist, break
    if not os.path.isfile(result_path): break
    result = pd.read_csv(result_path)
    # Add config index as a column
    result['Config Index'] = config_idx
    if results is None:
      results = result
    else:
      results = results.append(result, ignore_index=True)
    config_idx += total_combination
  
  return results

def merge_results(game, agent, result_label, mode):
  '''
  Show test and train results
  - Merge results
  - Sort results by result label in Test mode
  - Save results into a .csv file
  '''
  total_combination = get_total_combination(game, agent)
  for config_idx in range(1, total_combination+1):
    print(f'Merge {game}-{agent}-{mode} Results: {config_idx}')
    # Merge results
    results = merge_runs_results(game, agent, config_idx, total_combination, mode)
    if results is None:
      print(f'No {game}-{agent}-{mode} Results for {config_idx}')
      continue
    if mode == 'Test':
      # Sort test results by test result label
      results = results.sort_values(by=[result_label], ascending=[False])
    # Save results
    results_path = f'./{game}-{agent}-logs/{config_idx}/MergedResult-{mode}.csv'
    results.to_csv(results_path)

def plot_results(data, image_path, title, y_label, show=False):
  '''
  Plot results
  '''
  ax = sns.lineplot(x='Episode', y=y_label, hue='Agent', data=data)
  ax.set_title(title)
  ax.get_figure().savefig(image_path)
  if show:
    plt.show()
  plt.clf() # clear figure
  plt.cla() # clear axis
  plt.close() # close window

def plot_results_agent_index_list(game, agent_index_list, image_path, title, y_label, topk, show=False):
  '''
  Given game and agent-index pairs, plot the results
  '''
  mode = 'Train'
  results = None
  for agent_index in agent_index_list:
    agent, config_idx = agent_index.split('-')
    # Read test results
    test_results_path = f'./{game}-{agent}-logs/{config_idx}/MergedResult-Test.csv'
    if not os.path.isfile(test_results_path):
      print(f'No MergedResult-Test file for {game}-{agent}-{config_idx}!')
      continue
    test_results = pd.read_csv(test_results_path)
    # Get top k test results and config indexes
    test_results = test_results[:topk]
    # Get top k test results and config indexes
    config_idx_list = list(test_results['Config Index'])
    # Read train results
    train_results_path = f'./{game}-{agent}-logs/{config_idx}/MergedResult-{mode}.csv'
    if not os.path.isfile(train_results_path):
      print(f'No MergedResult-{mode} file for {game}-{agent}-{config_idx}!')
      continue
    # Get train results with config indexes in the config_idx_list only
    train_results = pd.read_csv(train_results_path)
    train_results = train_results.loc[train_results['Config Index'].isin(config_idx_list)]
    if results is None:
      results = train_results
    else:
      results = results.append(train_results, ignore_index=True)

  if results is not None:
    plot_results(results, image_path, title, y_label, show)
    
def show_results(game, agent, topk, result_label, mode):
  '''
  Show results based on mode
  '''
  results_list = [] 
  logs_path = f'./{game}-{agent}-logs/'
  images_path = f'./{game}-{agent}-images/'
  sweep_keys = get_sweep_keys(game, agent)
  total_combination = get_total_combination(game, agent)
  for config_idx in range(1, total_combination+1):
    print(f'Show {game}-{agent}-{mode} Results: {config_idx}')
    # Read test results
    test_results_path = f'./{game}-{agent}-logs/{config_idx}/MergedResult-Test.csv'
    if not os.path.isfile(test_results_path):
      print(f'No MergedResult-Test file for {game}-{agent}-{config_idx}!')
      continue
    test_results = pd.read_csv(test_results_path)
    # Get top k test results and config indexes
    test_results = test_results[:topk]
    if mode == 'Test':
      # Get mean and std of test results
      result_mean = test_results[result_label].mean()
      result_std = test_results[result_label].std(ddof=0)
      result_dict = {'Agent': agent,
                     'Game': game, 
                     'Config Index': config_idx, 
                     f'{result_label} (mean)': result_mean,
                     f'{result_label} (std)': result_std}
      # Expand test result dict from config dict
      log_dir = f'{config_idx}'
      config_file = os.path.join(logs_path, log_dir, 'config.json')
      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        for key in sweep_keys:
          result_dict[key] = config_dict[key]
      results_list.append(result_dict)
    elif mode == 'Train':
      # Get top k test results and config indexes
      config_idx_list = list(test_results['Config Index'])
      # Read train results
      train_results_path = f'./{game}-{agent}-logs/{config_idx}/MergedResult-{mode}.csv'
      if not os.path.isfile(train_results_path):
        print(f'No MergedResult-{mode} file for {game}-{agent}-{config_idx}!')
        continue
      train_results = pd.read_csv(train_results_path)
      # Get train results with config indexes in the config_idx_list only
      train_results = train_results.loc[train_results['Config Index'].isin(config_idx_list)]
      if train_results is not None:
        # Plot train results
        image_path = images_path + f'{config_idx}.png'
        plot_results(train_results, image_path, game, train_result_label, show=False)
      else:
        print(f'No {game}-{agent}-{mode} Results for {config_idx}')

  if mode == 'Test':
    # Sort by mean and std of test result label value
    results = pd.DataFrame(results_list)
    sorted_results = results.sort_values(by=[f'{result_label} (mean)', f'{result_label} (std)'], ascending=[False, False])
    # Save sorted test results into a .csv file
    sorted_results_path = f'./analysis/TestResults-{game}-{agent}-{mode}.csv'
    sorted_results.to_csv(sorted_results_path, index=False)

def unfinished_index(game, agent):
  '''
  Find unfinished config indexes based the existence of result-mode.csv file
  '''
  mode = 'Test'
  logs_path = f'./{game}-{agent}-logs/'
  print(f'{game}-{agent}:', end=' ')
  for log_dir in os.listdir(logs_path):
    if not os.path.isdir(os.path.join(logs_path, log_dir)):
      continue
    if not os.path.isfile(os.path.join(logs_path, log_dir, f'result-{mode}.csv')):
      print(log_dir, end=' ')
  print()

if __name__ == "__main__":
  game = 'Pixelcopter'
  agent = 'DDQN'
  topk = 2
  test_result_label = 'Average Return'
  train_result_label = 'Rolling Return'
  '''
  merge_results(game, agent, test_result_label, mode='Test')
  merge_results(game, agent, train_result_label, mode='Train')  
  show_results(game, agent, topk, test_result_label, mode='Test')
  show_results(game, agent, topk, train_result_label, mode='Train')
  '''
  image_path = './analysis/copter.png'
  agent_index_list = ['DQN-1', 'DDQN-2']
  plot_results_agent_index_list(game, agent_index_list, image_path, game, 'Rolling Return', topk, show=True)
  