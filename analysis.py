import os
from utils.plot import Plotter, unfinished_index

def maxmin_minatar():
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
    'exp': 'minatar_2',
    'merged': True,
    'x_label': 'Episode',
    'y_label': 'Average Return',
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'ci': None,
    'EMA': False,
    'loc': 'lower right',
    'sweep_keys': ['lr', 'target_networks_num'],
    'sort_by': ['Return (mean)', 'Return (se)'],
    'ascending': [False, True]
  }
  plotter = Plotter(cfg)
  title = 'MinAtar'
  
  plotter.merge_allIndex('Train')
  plotter.plot_results('Train', title)
  plotter.process_result('Train', get_process_result_dict)
  plotter.csv_results('Train', get_csv_result_dict)
  
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)

def maxmin_catcher():
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
    'exp': 'catcher',
    'merged': True,
    'x_label': 'Episode',
    'y_label': 'Average Return',
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'ci': None,
    'EMA': False,
    'loc': 'lower right',
    'sweep_keys': ['lr', 'target_networks_num'],
    'sort_by': ['Return (mean)', 'Return (se)'],
    'ascending': [False, True]
  }
  plotter = Plotter(cfg)
  title = 'Catcher'
  
  plotter.merge_allIndex('Train')
  plotter.plot_results('Train', title)
  plotter.process_result('Train', get_process_result_dict)
  plotter.csv_results('Train', get_csv_result_dict)
  
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)

def maxmin_copter():
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
    'x_label': 'Episode',
    'y_label': 'Average Return',
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'ci': None,
    'EMA': False,
    'loc': 'lower right',
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
  
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)

def maxmin_lunar():
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
    'exp': 'lunar',
    'merged': True,
    'x_label': 'Episode',
    'y_label': 'Average Return',
    'hue_label': 'Agent',
    'show': False,
    'imgType': 'png',
    'ci': None,
    'EMA': False,
    'loc': 'lower right',
    'sweep_keys': ['lr', 'target_networks_num'],
    'sort_by': ['Return (mean)', 'Return (se)'],
    'ascending': [False, True]
  }
  plotter = Plotter(cfg)
  title = 'LunarLander'
  
  plotter.merge_allIndex('Train')
  plotter.plot_results('Train', title)
  plotter.process_result('Train', get_process_result_dict)
  plotter.csv_results('Train', get_csv_result_dict)
  
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)

if __name__ == "__main__":
  #maxmin_minatar()
  #unfinished_index('minatar_2', 2)
  #maxmin_catcher()
  #maxmin_lunar()
  maxmin_copter()  