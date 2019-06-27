import os
import json
import pandas as pd
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt

from utils.helper import make_dir

def get_total_combination(game, agent, config_idx=1):
  '''
  Get total combination of configs
  '''
  config_file = f'./logs/{game}/{agent}/{config_idx}/config.json'
  assert os.path.isfile(config_file), f'No config file <{config_file}>!'
  with open(config_file, 'r') as f:
    config_dict = json.load(f)
  return config_dict['total_combinations']

def get_sweep_keys(game, agent):
  '''
  Open sweep config file and return sweep keys as a list
  '''
  sweep_config_file = f'./configs/{game}/{agent}.json'
  assert os.path.isfile(sweep_config_file), f'No sweep config file <{sweep_config_file}>!'
  sweep_keys = []
  with open(sweep_config_file, 'r') as f:
    config_dicts = json.load(f)
    for key, values in config_dicts.items():
      if len(values) > 1:
        sweep_keys.append(key)
  return sweep_keys

def merge_game_agent_index(game, agent, config_idx, total_combination, mode):
  '''
  Given game, agent and index, merge the results of multiple runs
  '''
  results = None
  while True:
    result_file = f'./logs/{game}/{agent}/{config_idx}/result_{mode}.csv'
    # If result file doesn't exist, break
    if not os.path.isfile(result_file):
      break
    # Read result file
    result = pd.read_csv(result_file)
    # Add config index as a column
    result['Config Index'] = config_idx
    results = result if results is None else results.append(result, ignore_index=True)
    config_idx += total_combination
  return results

def merge_game_agent_allIndex(game, agent, mode):
  '''
  Given game, agent, merge and save the results of multiple runs for all config indexes
  - Merge results
  - Sort results by result label in Test mode
  - Save results into a .csv file
  '''
  total_combination = get_total_combination(game, agent)
  for config_idx in range(1, total_combination+1):
    print(f'Merge {game}_{agent}_{mode} results: {config_idx}/{total_combination}')
    # Merge results
    results = merge_game_agent_index(game, agent, config_idx, total_combination, mode)
    if results is None:
      print(f'No {game}_{agent}_{mode} results for {config_idx}')
      continue
    # Save results
    results_file = f'./logs/{game}/{agent}/{config_idx}/result_{mode}_merged.csv'
    results.to_csv(results_file, index=False)

def sort_merged_test_result(game, agent):
  '''
  Sort merged test result by 'Average Return' and 'Return std'
  '''
  total_combination = get_total_combination(game, agent)
  for config_idx_ in range(1, total_combination+1):
    print(f'Sort {game}_{agent} merged test results: {config_idx_}/{total_combination}')
    new_results = None
    config_idx = config_idx_
    while True:
      # result = results.loc[results['Config Index']==config_idx]
      result_file = f'./logs/{game}/{agent}/{config_idx}/result_Test.csv'
      if not os.path.isfile(result_file): break
      result = pd.read_csv(result_file)
      new_result = pd.DataFrame([{'Game': game,
                                  'Agent': agent,
                                  'Config Index': config_idx,
                                  'Average Return': result['Return'].mean(),
                                  'Return std': result['Return'].std(ddof=0)}])
      new_results = new_result if new_results is None else new_results.append(new_result, ignore_index=True)
      config_idx += total_combination
    sorted_results = new_results.sort_values(by=['Average Return', 'Return std'], ascending=[False, True])
    # Save sorted results
    sorted_results_file = f'./logs/{game}/{agent}/{config_idx_}/result_Test_merged_sorted.csv'
    sorted_results.to_csv(sorted_results_file, index=False)    

def get_result(game, agent, config_idx, mode, merged, topKRun=0):
  '''
  Return: (merged) result 
  - merged == True: Return merged result for top k runs.
                    If topKRun == 0, return all merged result of all runs.
  - merged == False: Return unmerged result of one single run.
  '''
  if merged == False:
    result_file = f'./logs/{game}/{agent}/{config_idx}/result_{mode}.csv'
    assert os.path.isfile(result_file), f'No result file for <{result_file}>!'
    result = pd.read_csv(result_file)
  elif merged == True:
    if mode == 'Test': 
      result_file = f'./logs/{game}/{agent}/{config_idx}/result_{mode}_merged_sorted.csv'
      assert os.path.isfile(result_file), f'No result file for <{result_file}>!'
      result = pd.read_csv(result_file)
      if topKRun > 0:
        result = result[:topKRun]
    elif mode == 'Train':
      result_file = f'./logs/{game}/{agent}/{config_idx}/result_{mode}_merged.csv'
      assert os.path.isfile(result_file), f'No result file for <{result_file}>!'
      result = pd.read_csv(result_file)
      if topKRun > 0:
        # Read merged test results
        sorted_merged_test_result_file = f'./logs/{game}/{agent}/{config_idx}/result_Test_merged_sorted.csv'
        assert os.path.isfile(sorted_merged_test_result_file), f'No result file for <{sorted_merged_test_result_file}>!'
        sorted_merged_test_result = pd.read_csv(sorted_merged_test_result_file)
        # Get top k test results and config indexes
        sorted_merged_test_result = sorted_merged_test_result[:topKRun]
        config_idx_list = list(sorted_merged_test_result['Config Index'])
        # Get train results with config indexes in the config_idx_list only
        result = result.loc[result['Config Index'].isin(config_idx_list)]
  return result

def plot_vanilla(data, image_path, title, y_label, show=False):
  '''
  Plot results for any data (vanilla)
  '''
  ax = sns.lineplot(x='Episode', y=y_label, hue='Agent', data=data)
  ax.set_title(title)
  ax.get_figure().savefig(image_path)
  if show:
    plt.show()
  plt.clf() # clear figure
  plt.cla() # clear axis
  plt.close() # close window

def plot_game_agent_index(game, agent, config_idx, title, y_label, show, merged, topKRun=0):
  '''
  Mode: Train
  Func: Given game, agent and config index, plot (merged) train result
  - merged == True: Plot merged train result for top k runs.
                    If topKRun == 0, plot all merged train result of all runs.
  - merged == False: Plot unmerged train result of one single run. 
                     Note that topKRun==0 in this case.
  '''  
  # Get result
  result = get_result(game, agent, config_idx, 'Train', merged, topKRun)
  # Plot
  if merged:
    image_path = f'./logs/{game}/{agent}/{config_idx}/{config_idx}_merged_topKRun{topKRun}.png'
  else:
    image_path = f'./logs/{game}/{agent}/{config_idx}/{config_idx}.png'
  assert result is not None, 'No result!'
  plot_vanilla(result, image_path, title, y_label, show)

def plot_game_agent_indexList(game, agent, indexList, title, y_label, show, merged, topKRun=0):
  '''
  Mode: Train
  Func: Given game, agent and config index list, plot (merged) train result
  - merged == True: Plot merged train result for top k runs.
                    If topKRun == 0, plot all merged train result of all runs.
  - merged == False: Plot unmerged train result of one single run. 
                     Note that topKRun==0 in this case.
  '''
  results = None
  for config_idx in indexList:
    result = get_result(game, agent, config_idx, 'Train', merged, topKRun)
    # Modify "Agent" value in result for better visualization: add config index
    result['Agent'] = result['Agent'].map(lambda x: f'{x} {config_idx}')
    results = result if results is None else results.append(result, ignore_index=True)
  # Plot
  if merged:
    image_path = f'./logs/{game}/{agent}/0/indexList_merged_topKRun{topKRun}.png'
  else:
    image_path = f'./logs/{game}/{agent}/0/indexList.png'
  assert results is not None, 'No results!'
  plot_vanilla(results, image_path, title, y_label, show)

def plot_game_agentIndexList(game, agentIndexList, title, y_label, show, merged, topKRun=0):
  '''
  Mode: Train
  Func: Given game, agent-index list, plot (merged) train result
  - merged == True: Plot merged train result for top k runs.
                    If topKRun == 0, plot all merged train result of all runs.
  - merged == False: Plot unmerged train result of one single run. 
                     Note that topKRun==0 in this case.
  '''
  results = None
  for agent_index in agentIndexList:
    agent, config_idx = agent_index
    result = get_result(game, agent, config_idx, 'Train', merged, topKRun)
    # Modify "Agent" value in result for better visualization: add config index
    result['Agent'] = result['Agent'].map(lambda x: f'{x} {config_idx}')
    results = result if results is None else results.append(result, ignore_index=True)
  # Plot
  if merged:
    image_path = f'./logs/{game}/{agent}/0/agentIndexList_merged_topKRun{topKRun}.png'
  else:
    image_path = f'./logs/{game}/{agent}/0/agentIndexList.png'
  assert results is not None, 'No results!'
  plot_vanilla(results, image_path, title, y_label, show)
    
def show_results(game, agent, mode, result_label, show, topKRun=0):
  '''
  Merge: True
  Show results based on mode:
  - Train: Plot merged train result for all config indexes
  - Test: Generate a .csv file that store all merged test results
  '''

  total_combination = get_total_combination(game, agent)
  if mode == 'Test':
    results_list = []
    sweep_keys = get_sweep_keys(game, agent)
  for config_idx in range(1, total_combination+1):
    print(f'Show {game}_{agent}_{mode} results: {config_idx}/{total_combination}')
    result = get_result(game, agent, config_idx, mode, True, topKRun)
    if mode == 'Train':
      # Plot train results
      plot_game_agent_index(game, agent, config_idx, game, result_label, show, True, topKRun)
    elif mode == 'Test':
      # Get mean and std of test results
      result_dict = {'Agent': agent,
                     'Game': game, 
                     'Config Index': config_idx, 
                     f'{result_label} (mean)': result[result_label].mean(),
                     f'{result_label} (std)': result[result_label].std(ddof=0)}
      # Expand test result dict from config dict
      config_file = f'./logs/{game}/{agent}/{config_idx}/config.json'
      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        for key in sweep_keys:
          result_dict[key] = config_dict[key]
      results_list.append(result_dict)
  if mode == 'Test':
    make_dir(f'./logs/{game}/{agent}/0/')
    results = pd.DataFrame(results_list)
    # Sort by mean and std of test result label value
    sorted_results = results.sort_values(by=[f'{result_label} (mean)', f'{result_label} (std)'], ascending=[False, True])
    # Save sorted test results into a .csv file
    sorted_results_file = f'./logs/{game}/{agent}/0/TestResults_topKRun{topKRun}.csv'
    sorted_results.to_csv(sorted_results_file, index=False)

def unfinished_index(game, agent, runs):
  '''
  Find unfinished config indexes based on the existence of result_Test.csv file
  '''
  largest_config_idx = runs * get_total_combination(game, agent)
  print(f'{game}_{agent}:', end=' ')
  for config_idx in range(1, largest_config_idx + 1):
    result_file = f'./logs/{game}/{agent}/{config_idx}/result_Test.csv'
    if not os.path.isfile(result_file):
      print(config_idx, end=' ')
  print()

if __name__ == "__main__":
  game = 'Pixelcopter'
  title = game
  agent = 'DQN'
  show = True
  topKRun = 0
  test_result_label = 'Average Return'
  train_result_label = 'Rolling Return'
  
  merge_game_agent_allIndex(game, agent, mode='Test')
  merge_game_agent_allIndex(game, agent, mode='Train')
  sort_merged_test_result(game, agent)
  show_results(game, agent, 'Test', test_result_label, show, topKRun)
  show_results(game, agent, 'Train', train_result_label, show, topKRun)
  