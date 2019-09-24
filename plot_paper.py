import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt; plt.style.use('seaborn-ticks')
from matplotlib.ticker import FuncFormatter

from utils.helper import make_dir

# Set font family, bold, and font size
#font = {'family':'normal', 'weight':'normal', 'size': 12}
font = {'size': 15}
matplotlib.rc('font', **font)

class Plotter_paper(object):
  def __init__(self, cfg):
    cfg.setdefault('ci', None)
    self.x_label = cfg['x_label']
    self.y_label = cfg['y_label']
    self.show = cfg['show']
    self.imgType = cfg['imgType']
    self.ci = cfg['ci']
    self.loc = cfg['loc']
    make_dir('./figure_paper/')

  def get_result(self, exp, config_idx, mode):
    '''
    Given exp and config index, get the results
    '''
    xs, ys = None, None
    result_file = f'./logs/{exp}/{config_idx}/result_{mode}_merged.feather'  
    # Read result file
    result = pd.read_feather(result_file)
    xs = result[self.x_label].to_numpy()
    ys = result[self.y_label].to_numpy()
    # Compute x_mean, y_mean and y_ci
    x_mean, y_mean, y_ci = sorted(np.unique(xs)), [], []
    for x in x_mean:
      y_ = ys[xs == x]
      y_mean.append(np.mean(y_))
      if self.ci == 'sd':
        y_ci.append(np.std(y_, ddof=0))
      elif self.ci == 'se':
        y_ci.append(np.std(y_, ddof=0)/math.sqrt(len(y_)))

    y_mean, y_ci = np.array(y_mean), np.array(y_ci)
    return x_mean, y_mean, y_ci

def x_format(x, pos):
  #return '$%.1f$x$10^{6}$' % (x/1e6)
  return '%.1f' % (x/1e6)

cfg = {
  'x_label': 'Step',
  'y_label': 'Return',
  'show': False,
  'imgType': 'png',
  'ci': 'se',
  'x_format': x_format,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'loc': 'lower right'
}


def maxmin(exp_name):
  plotter = Plotter_paper(cfg)
  fig, ax = plt.subplots()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  label_list = ['DQN', 'DDQN', 'Averaged DQN', 'Maxmin DQN']
  color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'blue', 'green', 'orange', 'navy', 'yellow', 'purple']

  if 'copter' in exp_name:
    title = 'Pixelcopter'
    expIndexModeList = [('copter', 55, 'Train'), ('copter', 56, 'Train'), ('copter', 70, 'Train'), ('copter_maxmin', 13, 'Train')]
  elif 'catcher' in exp_name:
    title = 'Catcher'
    expIndexModeList = [('catcher', 37, 'Train'), ('catcher', 38, 'Train'), ('catcher_maxmin', 13, 'Train')]
  elif 'lunar' in exp_name:
    title = 'Lunarlander'
    expIndexModeList = [('lunar', 73, 'Train'), ('lunar', 74, 'Train'), ('lunar_maxmin', 10, 'Train')]
  elif 'minatar' in exp_name:
    title = 'MinAtar'
  else:
    title = exp_name
  #ax.set_title(title)
  
  # Draw
  for i in range(len(expIndexModeList)):
    exp, config_idx, mode = expIndexModeList[i]
    print(f'[{exp}]: Plot {mode} results: {config_idx}')
    x_mean, y_mean, y_ci = plotter.get_result(exp, config_idx, mode)
    plt.plot(x_mean, y_mean, linewidth=1.5, color=color_list[i], label=label_list[i])
    if cfg['ci'] in ['se', 'sd']:
      plt.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_list[i], alpha=0.5)
  
  # Set x and y axis
  ax.set_xlabel("Steps (x$10^{6}$)", fontsize=16)
  ax.set_ylabel('Return', fontsize=16, rotation='horizontal')
  ax.yaxis.set_label_coords(-0.18,0.5)
  #ax.xaxis.set_label_coords(0.0,0.5)
  ax.set_xlim(cfg['xlim']['min'], cfg['xlim']['max'])
  ax.set_ylim(cfg['ylim']['min'], cfg['ylim']['max'])
  ax.locator_params(nbins=5, axis='x')
  #ax.locator_params(nbins=5, axis='y')
  if not (cfg['x_format'] is None):
    ax.xaxis.set_major_formatter(FuncFormatter(cfg['x_format']))
  if not (cfg['y_format'] is None):
    ax.yaxis.set_major_formatter(FuncFormatter(cfg['y_format']))
  
  # Set legend
  ax.legend(loc=cfg['loc'], frameon=False, fontsize=16)
  
  # Adjust to show y label
  fig.subplots_adjust(left=0.2, bottom=0.14)
  
  # Save and show
  image_path = f'./figure_paper/{exp_name}.{cfg["imgType"]}'
  ax.get_figure().savefig(image_path)
  if cfg['show']:
    plt.show()
  plt.clf()   # clear figure
  plt.cla()   # clear axis
  plt.close() # close window

if __name__ == "__main__":
  maxmin('copter_maxmin')
  '''
  for exp in ['copter_maxmin', 'lunar_maxmin', 'catcher_maxmin']:
    maxmin(exp)
  '''