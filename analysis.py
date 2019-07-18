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
      'Return (mean)': result['Return'][-100:].mean()
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
    'exp': 'copter',
    'merged': True,
    'x_label': 'Step',
    'y_label': 'Average Return',
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'ci': 68,
    'EMA': True,
    'sweep_keys': ['lr', 'target_networks_num'],
    'sort_by': ['Return (mean)', 'Return (se)'],
    'ascending': [False, True]
  }
  plotter = Plotter(cfg)
  title = 'Pixelcopter'
  
  plotter.merge_allIndex('Train')
  plotter.plot_results('Train', title)
  plotter.process_result('Train', get_process_result_dict)
  plotter.csv_results('Train', get_csv_result_dict)

  '''
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)
  
  indexList = [1, 2, 3, 4]
  plotter.plot_indexList(indexList, 'Train', title)
  
  exp = 'test'
  expIndexList = [[exp, 4, 'Train'], [exp, 4, 'Test']]
  image_name = 'test'
  plotter.plot_expIndexList(expIndexList, change_hue_label, title, image_name)
  '''
if __name__ == "__main__":
  maxmin()