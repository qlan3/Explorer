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
  'sweep_keys': ['gradient_clip', 'state_normalizer'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp, runs=1):
  set_one_thread()
  cfg['exp'] = exp
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  plotter.csv_results('Train', get_csv_result_dict, get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Train', indexes='all')
  plotter.plot_results(mode='Test', indexes='all')
  # indexList = [11, 43, 15, 23, 31, 19, 22]
  # plotter.plot_indexList(indexList, 'Train', exp)
  
  envs = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "Ant-v2", "Reacher-v2"]
  indexes = {
    'onrpg': [31, 32, 33, 34, 35, 36],
    'ppo': [13, 14, 15, 16, 17, 18]
  }
  if exp == 'rpg_onrpg':
    for i in range(6):
      for mode in ['Train', 'Test']:
        expIndexModeList = [['rpg_onrpg', indexes['onrpg'][i], mode], ['rpg_ppo', indexes['ppo'][i], mode]]
        plotter.plot_expIndexModeList(expIndexModeList, f'{mode}_{envs[i]}')
  

if __name__ == "__main__":
  # unfinished_index('rpg_offrpg', runs=5)
  # memory_info('rpg_offrpg', runs=5)
  # time_info('rpg_offrpg', runs=5)
  # analyze('rpg_offrpg', runs=5)

  # unfinished_index('rpg_ppo', runs=10)
  # memory_info('rpg_ppo', runs=10)
  # time_info('rpg_ppo', runs=10)
  # analyze('rpg_ppo', runs=10)

  # unfinished_index('rpg_onrpg', runs=10)
  # memory_info('rpg_onrpg', runs=10)
  # time_info('rpg_onrpg', runs=10)
  analyze('rpg_onrpg', runs=10)