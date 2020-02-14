import os
import math
from utils.plotter import Plotter, unfinished_index
from utils.sweeper import time_info, memory_info
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

def maxmin(exp):
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
  else:
    title = exp
  
  plotter.plot_indexList(indexList=indexList, mode='Train', image_name=f'Train_{exp}')


if __name__ == "__main__":
  unfinished_index('copter_maxmin', range(1, 40+1))
  memory_info('copter_maxmin', runs=5)
  time_info('copter_maxmin', runs=5)
  maxmin('copter_maxmin')
  
  ''' The first index is the best.
  - copter: 
      DQN: 55,37,73
      DDQN: 56,38,74
      AveragedDQN: 70,66,68,54
      MaxminDQN: 49,61,63,65,67,69
  - copter_maxmin: 13
  - lunar:
      DQN: 73,55,37
      DDQN: 74,56,38
      AveragedDQN: 46,54,52,64,60,86,36,72,62
      MaxminDQN: 31,53,35,51,47,89,87,67
  - lunar_maxmin: 10
  - catcher: 
      DQN: 37,55,73
      DDQN: 38,56,74
      AveragedDQN: 42,40,44,58,64,88
      MaxminDQN: 45,53,39,47
  - catcher_maxmin: 13
  - Space_invaders: 
      DQN: 184,274
      DDQN: 189,279
      AveragedDQN: 229,209,219
      MaxminDQN: 214,234,224,254,244
  - Seaquest: 
      DQN: 190
      DDQN: 185
      AveragedDQN: 260,270,250
      MaxminDQN: 215,225,235,245
  - Breakout: 
      DQN: 92,182,2
      DDQN: 97,367
      AveragedDQN: 87,37,67,47,77
      MaxminDQN: 172,162,142,132
  - Asterix: 
      DQN: 271
      DDQN: 276
      AveragedDQN: 296,316,306
      MaxminDQN: 261,291,331,311,241
  '''