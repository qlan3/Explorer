import os
from utils.plot import Plotter

def maxmin():
  def change_hue_label(exp, hue_label, config_idx, mode):
    return f'[{exp}]-{mode} {hue_label} {config_idx}'

  def get_process_result_dict(result, config_idx):
    result_dict = {
      'Env': result['Env'][0],
      'Agent': result['Agent'][0],
      'Config Index': config_idx,
      'Return (mean)': result['Return'][-1000:].mean()
    }
    return result_dict
  
  def get_csv_result_dict(result, config_idx):
    result_dict = {
      'Env': result['Env'][0],
      'Agent': result['Agent'][0],
      'Config Index': config_idx,
      'Return (mean)': result['Return (mean)'].mean(),
      'Return (se)': result['Return (mean)'].sem(ddof=0)
      }
    return result_dict

  cfg = {
    'exp': 'atari_ram_3',
    'merged': True,
    'x_label': 'Step',
    'y_label': 'Average Return',
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'ci': 68,
    'EMA': True,
    'sweep_keys': ['hidden_layers', 'lr'],
    'sort_by': ['Return (mean)', 'Return (se)'],
    'ascending': [False, True]
  }
  plotter = Plotter(cfg)
  title = 'Atari Ram'
  
  plotter.merge_allIndex('Train')
  plotter.plot_results('Train', title)
  plotter.process_result('Train', get_process_result_dict)
  plotter.csv_results('Train', get_csv_result_dict)
  
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)

  '''
  indexList = [10, 22, 16]
  title = 'Qbert-ram-v4'
  plotter.plot_indexList(indexList, 'Train', title)

  exp = 'atari_ram_2'
  expIndexList = [[exp, 20, 'Train'], [exp, 2, 'Train'], [exp, 8, 'Train']]
  title = 'Enduro-ram-v4'
  image_name = 'Enduro-ram-v4'
  plotter.plot_expIndexList(expIndexList, change_hue_label, title, image_name) 
  '''
  
if __name__ == "__main__":
  maxmin()