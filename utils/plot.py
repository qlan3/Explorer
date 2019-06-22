import os
import json
import pandas as pd
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt


def get_total_combination(env, agent, config_idx=1):
  logs_path = f'./{env}-{agent}-logs/'
  log_dir = f'{config_idx}'
  assert os.path.isdir(os.path.join(logs_path, log_dir)), f'No log dir <{log_dir}>!'
  config_path = os.path.join(logs_path, log_dir, 'config.json')
  with open(config_path, 'r') as f:
    config_dict = json.load(f)
  return config_dict['total_combinations']

def get_sweep_keys(env, agent):
  '''
  Open sweep config file and return sweep keys as a list
  '''
  sweep_config_file = f'./configs/{env}-{agent}.json'
  sweep_keys = []
  with open(sweep_config_file, 'r') as f:
    config_dicts = json.load(f)
    for key, values in config_dicts.items():
      if len(values) > 1:
        sweep_keys.append(key)
  
  return sweep_keys

def merge_result(env, agent, config_idx, total_combination, mode='Train'):
  '''
  Merge single run result into multiple runs results
  '''
  logs_path = f'./{env}-{agent}-logs/'
  flag = False
  results = None
  while True:
    log_dir = f'{config_idx}'
    result_path = os.path.join(logs_path, log_dir, f'result-{mode}.csv')
    # If result file doesn't exist, break
    if not os.path.isfile(result_path): break
    result = pd.read_csv(result_path)
    if flag == False:
      results = result
      flag = True
    else:
      results = results.append(result, ignore_index=True)
    config_idx += total_combination
  
  return results

def plot_results(data, image_path, title, y_label, show):
  ax = sns.lineplot(x='Episode', y=y_label, hue='Agent', data=data)
  ax.set_title(title)
  ax.get_figure().savefig(image_path)
  if show:
    plt.show()
  plt.clf() # clear figure
  plt.cla() # clear axis
  plt.close() # close window

def plot_game_agent_results(game, agent, logs_dir, y_label, show=True):
  # TODO
  return

def plot_game_agents_results():
  # TODO
  return

def tb_plot_agent_results(tb_logs_dir):
  # TODO
  return

def show_train_results(env, agent, result_label='Rolling Return', mode='Train', show=False):
  '''
  1. Merge train results
  2. Plot the merged results
  '''
  images_path = f'./{env}-{agent}-images/'
  total_combination = get_total_combination(env, agent)
  for config_idx in range(1, total_combination+1):
    print(f'Show {mode} Results: {config_idx}')
    # Merge
    train_results = merge_result(env, agent, config_idx, total_combination, mode='Train')
    if train_results is None:
      print(f'No {mode} Results for {config_idx}')
      break
    # Plot
    image_path = images_path + f'{config_idx}.png'
    plot_results(train_results, image_path, env, result_label, show)
    train_results = None

def show_test_result(env, agent, result_label='Average Return', mode='Test'):
  '''
  1. Merge test results
  2. Get mean and std of test results
  3. Expand result dict from config dict
  4. Sort according to mean and std of result label value
  5. Save sorted results into a .csv file
  '''
  results_list = []
  logs_path = f'./{env}-{agent}-logs/'
  test_results_path = './analysis/'
  sweep_keys = get_sweep_keys(env, agent)
  total_combination = get_total_combination(env, agent)
  for config_idx in range(1, total_combination+1):
    print(f'Show {mode} Results: {config_idx}')
    # Merge
    test_results = merge_result(env, agent, config_idx, total_combination, mode)
    if test_results is None:
      print(f'No {mode} Results for {config_idx}')
      break
    # Get mean and std of test results
    result_mean = test_results[result_label].mean()
    result_std = test_results[result_label].std(ddof=0)
    result_dict = {'Agent': agent,
                   'Env': env, 
                   'Config Index': config_idx, 
                   f'{result_label} (mean)': result_mean,
                   f'{result_label} (std)': result_std}
    # Expand result dict from config dict
    log_dir = f'{config_idx}'
    config_file = os.path.join(logs_path, log_dir, 'config.json')
    with open(config_file, 'r') as f:
      config_dict = json.load(f)
      for key in sweep_keys:
        result_dict[key] = config_dict[key]
    results_list.append(result_dict)
  # Sort according to mean and std of result label value
  results = pd.DataFrame(results_list)
  sorted_results = results.sort_values(by=[f'{result_label} (mean)', f'{result_label} (std)'], ascending=[False, False])
  # Save sorted results into a .csv file
  sorted_results_path = test_results_path + f'TestResults-{env}-{agent}-{mode}.csv'
  sorted_results.to_csv(sorted_results_path, index=False)


if __name__ == "__main__":
  for env in ['Pixelcopter']:
    for agent in ['DQN']:
      show_train_results(env, agent, result_label='Rolling Return', mode='Train', show=False)
      show_test_result(env, agent, result_label='Average Return', mode='Test')