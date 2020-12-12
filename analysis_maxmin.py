import os
import math
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info
from utils.helper import set_one_thread


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
  'exp': 'exp_name',
  'merged': True,
  'x_label': 'Step',
  'y_label': 'Average Return',
  'hue_label': 'Agent',
  'show': False,
  'imgType': 'png',
  'ci': 'se',
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'lower right',
  'sweep_keys': ['lr', 'target_networks_num'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 5
}

def analyze(exp):
  set_one_thread()
  cfg['exp'] = exp
  plotter = Plotter(cfg)


  plotter.csv_results('Train', get_csv_result_dict, get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Train', indexes='all')

  if 'copter' in exp:
    title = 'Pixelcopter'
    indexList = [55, 56, 70, 29]
  elif 'catcher' in exp:
    title = 'Catcher'
    indexList = [37, 38, 42, 29]
  elif 'lunar' in exp:
    title = 'Lunarlander'
    indexList = [73, 74, 46, 41]
  elif 'minatar' in exp:
    title = 'MinAtar'
    indexList = []
  else:
    title = exp
    indexList = []
  
  plotter.plot_indexList(indexList=indexList, mode='Train', image_name=f'Train_{exp}')


if __name__ == "__main__":
  unfinished_index('catcher', runs=5)
  memory_info('catcher', runs=5)
  time_info('catcher', runs=5)
  analyze('catcher')