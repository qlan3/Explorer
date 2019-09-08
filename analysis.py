import os
from utils.plot import Plotter, unfinished_index

def x_format(x, pos):
  return '$%.1f$x$10^{6}$' % (x/1e6)

def change_hue_label(exp, hue_label, config_idx, mode):
  return f'[{exp}]-{mode} {hue_label} {config_idx}'

def change_hue_label_paper(exp, hue_label, config_idx, mode):
  if 'Maxmin' in hue_label:
    return 'Maxmin DQN'
  else:
    return hue_label

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
  'ci': 'sd',
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'EMA': True,
  'loc': 'lower right',
  'sweep_keys': ['lr', 'target_networks_num'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 20
}

def maxmin(exp):
  cfg['exp'] = exp
  plotter = Plotter(cfg)
  if 'copter' in exp:
    title = 'Pixelcopter'
    expIndexModeList = [('copter', 55, 'Train'), ('copter', 56, 'Train'), ('copter_maxmin', 13, 'Train')]
  elif 'catcher' in exp:
    title = 'Catcher'
    expIndexModeList = [('catcher', 37, 'Train'), ('catcher', 38, 'Train'), ('catcher_maxmin', 13, 'Train')]
  elif 'lunar' in exp:
    title = 'Lunarlander'
    expIndexModeList = [('lunar', 73, 'Train'), ('lunar', 74, 'Train'), ('lunar_maxmin', 10, 'Train')]
  elif 'minatar' in exp:
    title = 'MinAtar'
  else:
    title = exp
  
  plotter.merge_allIndex('Train')
  plotter.plot_results('Train', title)
  plotter.process_result('Train', get_process_result_dict)
  plotter.csv_results('Train', get_csv_result_dict)
  '''
  plotter.merge_allIndex('Test')
  plotter.plot_results('Test', title)
  plotter.process_result('Test', get_process_result_dict)
  plotter.csv_results('Test', get_csv_result_dict)
  '''
  # plotter.ci = 68
  #plotter.plot_expIndexModeList(expIndexModeList, change_hue_label_paper, title, f'Train_{exp}')

if __name__ == "__main__":
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
  - MinAtar: 
      DQN:
      DDQN:
      AveragedDQN:
      MaxminDQN:
  '''
  #for exp in ['copter_maxmin', 'lunar_maxmin', 'catcher_maxmin']:
    #maxmin(exp)
  maxmin('test')