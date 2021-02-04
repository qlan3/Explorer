import os
import math
from utils.plotter import Plotter
from utils.sweeper import unfinished_index, time_info, memory_info
from utils.helper import set_one_thread


def get_process_result_dict(result, config_idx, mode='Train'):
  result_dict = {
    'Env': result['Env'][0],
    'Agent': result['Agent'][0],
    'Config Index': config_idx,
    'Return (mean)': result['Return'][-100:].mean() if mode=='Train' else result['Return'][-5:].mean()
  }
  return result_dict

def get_csv_result_dict(result, config_idx, mode='Train'):
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
  'sweep_keys': [],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  set_one_thread()
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Test', indexes='all')
  
  envs = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "Ant-v2", "Reacher-v2"]
  indexes = {
    'ppo': [1, 2, 3, 4, 5, 6],
    'rpg': [7, 8, 9, 10, 11, 12]
  }
  if exp == 'rpg':
    for i in range(6):
      for mode in ['Test']:
        expIndexModeList = [['rpg', indexes['ppo'][i], mode], ['rpg', indexes['rpg'][i], mode]]
        plotter.plot_expIndexModeList(expIndexModeList, f'{mode}_{envs[i]}')
  

if __name__ == "__main__":
  unfinished_index('rpg', runs=30)
  memory_info('rpg', runs=30)
  time_info('rpg', runs=30)
  analyze('rpg', runs=30)