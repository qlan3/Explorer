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
  'sweep_keys': ['lr', 'name'],
  'sort_by': ['Return (mean)', 'Return (se)'],
  'ascending': [False, True],
  'runs': 1
}

def analyze(exp):
  set_one_thread()
  cfg['exp'] = exp
  plotter = Plotter(cfg)


  plotter.csv_results('Train', get_csv_result_dict, get_process_result_dict)
  # plotter.csv_results('Test', get_csv_result_dict, get_process_result_dict)
  plotter.plot_results(mode='Train', indexes='all')


if __name__ == "__main__":
  # unfinished_index('mujoco_a2c')
  memory_info('mujoco_a2c')
  time_info('mujoco_a2c')
  analyze('mujoco_a2c')

  memory_info('mujoco_ddpg')
  time_info('mujoco_ddpg')
  analyze('mujoco_ddpg')

  memory_info('mujoco_ppo')
  time_info('mujoco_ppo')
  analyze('mujoco_ppo')

  memory_info('mujoco_sac')
  time_info('mujoco_sac')
  analyze('mujoco_sac')

  memory_info('mujoco_td3')
  time_info('mujoco_td3')
  analyze('mujoco_td3')