import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt; plt.style.use('seaborn-ticks')
from matplotlib.ticker import FuncFormatter
# Avoid Type 3 fonts: http://phyletica.org/matplotlib-fonts/
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# Set font family, bold, and font size
#font = {'family':'normal', 'weight':'normal', 'size': 12}
font = {'size': 15}
matplotlib.rc('font', **font)

from utils.helper import make_dir
from utils.plotter import read_file, get_total_combination, symmetric_ema


class Plotter(object):
  def __init__(self, cfg):
    cfg.setdefault('ci', None)
    self.x_label = cfg['x_label']
    self.y_label = cfg['y_label']
    self.show = cfg['show']
    self.imgType = cfg['imgType']
    self.ci = cfg['ci']
    self.runs = cfg['runs']
    make_dir('./figures/')

  def get_result(self, exp, config_idx, mode):
    '''
    Given exp and config index, get the results
    '''
    total_combination = get_total_combination(exp)
    result_list = []
    for _ in range(self.runs):
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      # If result file exist, read and merge
      result = read_file(result_file)
      if result is not None:
        # Add config index as a column
        result['Config Index'] = config_idx
        result_list.append(result)
      config_idx += total_combination

    # Do symmetric EMA (exponential moving average)
    # Get x's and y's in form of numpy arries
    xs, ys = [], []
    for result in result_list:
      xs.append(result[self.x_label].to_numpy())
      ys.append(result[self.y_label].to_numpy())
    # Do symetric EMA to get new x's and y's
    low  = max(x[0] for x in xs)
    high = min(x[-1] for x in xs)
    n = min(len(x) for x in xs)
    for i in range(len(xs)):
      new_x, new_y, _ = symmetric_ema(xs[i], ys[i], low, high, n)
      result_list[i] = result_list[i][:n]
      result_list[i].loc[:, self.x_label] = new_x
      result_list[i].loc[:, self.y_label] = new_y

    ys = []
    for result in result_list:
      ys.append(result[self.y_label].to_numpy())
    # Compute x_mean, y_mean and y_ci
    ys = np.array(ys)
    x_mean = result_list[0][self.x_label].to_numpy()
    y_mean = np.mean(ys, axis=0)
    if self.ci == 'sd':
      y_ci = np.std(ys, axis=0, ddof=0)  
    elif self.ci == 'se':
      y_ci = np.std(ys, axis=0, ddof=0)/math.sqrt(len(ys))
      
    return x_mean, y_mean, y_ci

def x_format(x, pos):
  #return '$%.1f$x$10^{6}$' % (x/1e6)
  return '%.1f' % (x/1e6)

cfg = {
  'x_label': 'Step',
  'y_label': 'Average Return',
  'show': False,
  'imgType': 'png',
  'ci': 'se',
  'x_format': None,
  'y_format': None,
  'xlim': {'min': None, 'max': None},
  'ylim': {'min': None, 'max': None},
  'runs': 10,
  'loc': 'lower right'
}


def learning_curve(exp, runs=1):
  cfg['runs'] = runs
  plotter = Plotter(cfg)

  label_list = ['PPO', 'RPG']
  color_list = ['tab:red', 'tab:blue']
  envs = ["HalfCheetah-v2", "Hopper-v2", "Walker2d-v2", "Swimmer-v2", "Ant-v2", "Reacher-v2"]
  indexes = {
    'PPO': [1, 2, 3, 4, 5, 6],
    'RPG': [7, 8, 9, 10, 11, 12]
  }
  
  for i in range(len(envs)):
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Plot
    env = envs[i]
    for j in range(len(label_list)):
      agent = label_list[j]
      config_idx = indexes[agent][i]
      print(f'[{exp}]: Plot Test results: {config_idx}')
      x_mean, y_mean, y_ci = plotter.get_result(exp, config_idx, 'Test')
      plt.plot(x_mean, y_mean, linewidth=1.5, color=color_list[j], label=agent)
      if cfg['ci'] in ['se', 'sd']:
        plt.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, facecolor=color_list[j], alpha=0.5)  
    # Set x and y axis
    # ax.set_xlabel("Steps (x$10^{6}$)", fontsize=16)
    # ax.set_ylabel('Average Return', fontsize=16, rotation='horizontal')
    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel('Average Return', fontsize=16)
    plt.yticks(size=11)
    plt.xticks(size=11)
    # ax.yaxis.set_label_coords(-0.18,0.5)
    # ax.xaxis.set_label_coords(0.0,0.5)
    # ax.set_xlim(cfg['xlim']['min'], cfg['xlim']['max'])
    # ax.set_ylim(cfg['ylim']['min'], cfg['ylim']['max'])
    # ax.locator_params(nbins=5, axis='x')
    # ax.locator_params(nbins=5, axis='y')
    # if not (cfg['x_format'] is None):
    #   ax.xaxis.set_major_formatter(FuncFormatter(cfg['x_format']))
    # if not (cfg['y_format'] is None):
    #   ax.yaxis.set_major_formatter(FuncFormatter(cfg['y_format']))  
    # # Set legend
    ax.legend(loc=cfg['loc'], frameon=False, fontsize=16)
    # Adjust to show y label
    # fig.subplots_adjust(left=0.2, bottom=0.14)
    # Save and show
    image_path = f'./figures/{env}.{cfg["imgType"]}'
    ax.get_figure().savefig(image_path)
    if cfg['show']:
      plt.show()
    plt.clf()   # clear figure
    plt.cla()   # clear axis
    plt.close() # close window

if __name__ == "__main__":
  learning_curve('RPG', 30)