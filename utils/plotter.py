import os
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks"); sns.set_context("paper") #sns.set_context("talk")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from utils.helper import make_dir
from utils.sweeper import Sweeper


class Plotter(object):
  def __init__(self, cfg):
    # Set default value for symmetric EMA (exponential moving average)
    # Note that EMA only works when merged is True
    cfg.setdefault('EMA', False)
    cfg.setdefault('ci', None)
    # Copy parameters
    self.exp = cfg['exp']
    self.merged = cfg['merged']
    self.x_label = cfg['x_label']
    self.y_label = cfg['y_label']
    self.hue_label = cfg['hue_label']
    self.show = cfg['show']
    self.imgType = cfg['imgType']
    self.ci = cfg['ci']
    self.EMA = cfg['EMA']
    self.sweep_keys = cfg['sweep_keys']
    self.sort_by = cfg['sort_by']
    self.ascending = cfg['ascending']
    self.loc = cfg['loc']
    self.runs = cfg['runs']
    # Get total combination of configurations
    self.total_combination = get_total_combination(self.exp)

  def merge_index(self, config_idx, mode, processed, exp=None):
    '''
    Given exp and config index, merge the results of multiple runs
    '''
    if exp is None:
      exp = self.exp
    result_list = []
    for _ in range(self.runs):
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      # If result file exist, read and merge
      result = read_file(result_file)
      if result is not None:
        # Add config index as a column
        result['Config Index'] = config_idx
        result_list.append(result)
      config_idx += self.total_combination
    
    if len(result_list) == 0:
      return None
    
    # Do symmetric EMA (exponential moving average) only
    # when we want the original data (i.e. no processed)
    if (self.EMA) and (processed == False):
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
    else:
      # Cut off redundant results
      n = min(len(result) for result in result_list)
      for i in range(len(result_list)):
        result_list[i] = result_list[i][:n]

    return result_list

  def get_result(self, exp, config_idx, mode, get_process_result_dict=None):
    '''
    Return: (merged, processed) result 
    - if (merged == True) or (get_process_result_dict is not None):
        Return a list of (processed) result for all runs.
    - if (merged == False):
        Return unmerged result of one single run in a list.
    '''
    if get_process_result_dict is not None:
      processed = True
    else:
      processed = False
    
    if self.merged == True or processed == True:
      # Check mode
      if not (mode in ['Train', 'Valid', 'Test', 'Dynamic']):
        return None
      # Merge results
      print(f'[{exp}]: Merge {mode} results: {config_idx}/{self.total_combination}')
      result_list = self.merge_index(config_idx, mode, processed, exp)
      if result_list is None:
        print(f'[{exp}]: No {mode} results for {config_idx}')
        return None
      # Process result
      if processed:
        print(f'[{exp}]: Process {mode} results: {config_idx}/{self.total_combination}')
        for i in range(len(result_list)):
          new_result = get_process_result_dict(result_list[i], config_idx)
          result_list[i] = new_result
      return result_list
    else:
      result_file = f'./logs/{exp}/{config_idx}/result_{mode}.feather'
      result = read_file(result_file)
      if result is None:
        return None
      else:
        return [result]

  def plot_vanilla(self, data, image_path):
    '''
    Plot results for data:
      data = [result_1_list, result_2_list, ...]
      result_i_list = [result_run_1, result_run_2, ...]
      result_run_i is a Dataframe
    '''
    fig, ax = plt.subplots()
    for i in range(len(data)):
      # Convert to numpy array
      ys = []
      for result in data[i]:
        ys.append(result[self.y_label].to_numpy())
      # Compute x_mean, y_mean and y_ci
      ys = np.array(ys)
      x_mean = data[i][0][self.x_label].to_numpy()
      y_mean = np.mean(ys, axis=0)
      if self.ci == 'sd':
        y_ci = np.std(ys, axis=0, ddof=0)  
      elif self.ci == 'se':
        y_ci = np.std(ys, axis=0, ddof=0)/math.sqrt(len(ys))
      # Plot
      plt.plot(x_mean, y_mean, linewidth=1.0, label=data[i][0][self.hue_label][0])
      if self.ci in ['sd', 'se']:
        plt.fill_between(x_mean, y_mean - y_ci, y_mean + y_ci, alpha=0.5)
    
    # ax.set_title(title)
    ax.legend(loc=self.loc)
    ax.set_xlabel(self.x_label)
    ax.set_ylabel(self.y_label)
    ax.get_figure().savefig(image_path)
    if self.show:
      plt.show()
    plt.clf()   # clear figure
    plt.cla()   # clear axis
    plt.close() # close window

  def plot_indexList(self, indexList, mode, image_name):
    '''
    Func: Given (config index) list and mode
    - merged == True: plot merged result for all runs.
    - merged == False: plot unmerged result of one single run. 
    '''
    expIndexModeList = []
    for x in indexList:
      expIndexModeList.append([self.exp, x ,mode])
    self.plot_expIndexModeList(expIndexModeList, image_name)

  def plot_indexModeList(self, indexModeList, image_name):
    '''
    Func: Given (config index, mode) list
    - merged == True: plot merged result for all runs.
    - merged == False: plot unmerged result of one single run. 
    '''
    expIndexModeList = []
    for x in indexModeList:
      expIndexModeList.append([self.exp] + x)
    self.plot_expIndexModeList(expIndexModeList, image_name)

  def plot_expIndexModeList(self, expIndexModeList, image_name):
    '''
    Func: Given (exp, config index, mode) list
    - merged == True: plot merged result for all runs.
    - merged == False: plot unmerged result of one single run.
    '''
    # Get results
    results = []
    for exp, config_idx, mode in expIndexModeList:
      print(f'[{exp}]: Plot {mode} results: {config_idx}')
      result_list = self.get_result(exp, config_idx, mode)
      if result_list is None:
        continue
      # Modify `hue_label` value in result_list for better visualization
      for i in range(len(result_list)):
        result_list[i][self.hue_label] = result_list[i][self.hue_label].map(lambda x: f'[{exp}] {mode} {x} {config_idx}')
      results.append(result_list)
    
    make_dir(f'./logs/{self.exp}/0/')
    # Plot
    if self.merged:
      image_path = f'./logs/{self.exp}/0/{image_name}_merged.{self.imgType}'
    else:
      image_path = f'./logs/{self.exp}/0/{image_name}.{self.imgType}'
    self.plot_vanilla(results, image_path)

  def plot_results(self, mode, indexes='all'):
    '''
    Plot merged result for all config indexes
    '''
    if indexes == 'all':
      if self.merged:
        indexes = range(1, self.total_combination+1)
      else:
        indexes = range(1, self.total_combination*self.runs+1)

    for config_idx in indexes:
      print(f'[{self.exp}]: Plot {mode} results: {config_idx}/{self.total_combination}')
      # Get result
      result_list = self.get_result(self.exp, config_idx, mode)
      if result_list is None:
        continue
      # Plot
      if self.merged:
        image_path = f'./logs/{self.exp}/{config_idx}/{mode}_{self.y_label}_merged.{self.imgType}'
      else:
        image_path = f'./logs/{self.exp}/{config_idx}/{mode}_{self.y_label}.{self.imgType}'
      self.plot_vanilla([result_list], image_path)

  def csv_results(self, mode, get_csv_result_dict, get_process_result_dict):
    '''
    Show results: generate a *.csv file that store all merged results
    '''
    new_result_list = []
    for config_idx in range(1, self.total_combination+1):
      print(f'[{self.exp}]: CSV {mode} results: {config_idx}/{self.total_combination}')
      result_list = self.get_result(self.exp, config_idx, mode, get_process_result_dict)
      if result_list is None:
        continue
      result = pd.DataFrame(result_list)
      # Get test results dict
      result_dict = get_csv_result_dict(result, config_idx)
      # Expand test result dict from config dict
      config_file = f'./logs/{self.exp}/{config_idx}/config.json'
      with open(config_file, 'r') as f:
        config_dict = json.load(f)
        for key in self.sweep_keys:
          result_dict[key] = find_key_value(config_dict, key)
      new_result_list.append(result_dict)

    if len(new_result_list) == 0:
      print(f'[{self.exp}]: No {mode} results')
      return
    make_dir(f'./logs/{self.exp}/0/')
    results = pd.DataFrame(new_result_list)
    # Sort by mean and ste of test result label value
    sorted_results = results.sort_values(by=self.sort_by, ascending=self.ascending)
    # Save sorted test results into a .feather file
    sorted_results_file = f'./logs/{self.exp}/0/{mode}_results.csv'
    sorted_results.to_csv(sorted_results_file, index=False)



def one_sided_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=0.0):
  ''' Copy from baselines.common.plot_util
  Functionality:
    perform one-sided (causal) EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
  Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
  Returns:
    tuple sum_ys, count_ys where
      xs                  - array with new x grid
      ys                  - array of EMA of y at each point of the new x grid
      count_ys            - array of EMA of y counts at each point of the new x grid
  '''

  low = xolds[0] if low is None else low
  high = xolds[-1] if high is None else high

  assert xolds[0] <= low, 'low = {} < xolds[0] = {} - extrapolation not permitted!'.format(low, xolds[0])
  assert xolds[-1] >= high, 'high = {} > xolds[-1] = {}  - extrapolation not permitted!'.format(high, xolds[-1])
  assert len(xolds) == len(yolds), 'length of xolds ({}) and yolds ({}) do not match!'.format(len(xolds), len(yolds))

  xolds, yolds = xolds.astype('float64'), yolds.astype('float64')
  luoi = 0 # last unused old index
  sum_y = 0.
  count_y = 0.
  xnews = np.linspace(low, high, n)
  decay_period = (high - low) / (n - 1) * decay_steps
  interstep_decay = np.exp(- 1. / decay_steps)
  sum_ys = np.zeros_like(xnews)
  count_ys = np.zeros_like(xnews)
  for i in range(n):
    xnew = xnews[i]
    sum_y *= interstep_decay
    count_y *= interstep_decay
    while True:
      if luoi >= len(xolds): break
      xold = xolds[luoi]
      if xold <= xnew:
        decay = np.exp(- (xnew - xold) / decay_period)
        sum_y += decay * yolds[luoi]
        count_y += decay
        luoi += 1
      else: break
    sum_ys[i] = sum_y
    count_ys[i] = count_y

  ys = sum_ys / count_ys
  ys[count_ys < low_counts_threshold] = np.nan
  return xnews, ys, count_ys

def symmetric_ema(xolds, yolds, low=None, high=None, n=512, decay_steps=1.0, low_counts_threshold=0.0):
  ''' Copy from baselines.common.plot_util
  Functionality:
    Perform symmetric EMA (exponential moving average)
    smoothing and resampling to an even grid with n points.
    Does not do extrapolation, so we assume
    xolds[0] <= low && high <= xolds[-1]
  Arguments:
    xolds: array or list  - x values of data. Needs to be sorted in ascending order
    yolds: array of list  - y values of data. Has to have the same length as xolds
    low: float            - min value of the new x grid. By default equals to xolds[0]
    high: float           - max value of the new x grid. By default equals to xolds[-1]
    n: int                - number of points in new x grid
    decay_steps: float    - EMA decay factor, expressed in new x grid steps.
    low_counts_threshold: float or int
                          - y values with counts less than this value will be set to NaN
  Returns:
    tuple sum_ys, count_ys where
      xs        - array with new x grid
      ys        - array of EMA of y at each point of the new x grid
      count_ys  - array of EMA of y counts at each point of the new x grid

  '''
  xs, ys1, count_ys1 = one_sided_ema(xolds, yolds, low, high, n, decay_steps, low_counts_threshold)
  _,  ys2, count_ys2 = one_sided_ema(-xolds[::-1], yolds[::-1], -high, -low, n, decay_steps, low_counts_threshold)
  ys2 = ys2[::-1]
  count_ys2 = count_ys2[::-1]
  count_ys = count_ys1 + count_ys2
  ys = (ys1 * count_ys1 + ys2 * count_ys2) / count_ys
  ys[count_ys < low_counts_threshold] = np.nan
  xs = [int(x) for x in xs]
  return xs, ys, count_ys

def moving_average(values, window):
  '''
  Smooth values by doing a moving average
  :param values: (numpy array)
  :param window: (int)
  :return: (numpy array)
  '''
  weights = np.repeat(1.0, window) / window
  return np.convolve(values, weights, 'valid')
  
def get_total_combination(exp):
  '''
  Get total combination of experiment configuration
  '''
  config_file = f'./configs/{exp}.json'
  assert os.path.isfile(config_file), f'[{exp}]: No config file <{config_file}>!'
  sweeper = Sweeper(config_file)
  return sweeper.config_dicts['num_combinations']


def find_key_value(config_dict, key):
  '''
  Find key value in config dict recursively
  '''
  for k, v in config_dict.items():
    if k == key:
      return config_dict[k]
    elif type(v) == dict:
      value = find_key_value(v, key)
      if value is not '/':
        return value
  return '/'


def read_file(result_file):
  if not os.path.isfile(result_file):
    print(f'[No such file <{result_file}>')
    return None
  result = pd.read_feather(result_file)
  if result is None:
    print(f'No result in file <{result_file}>')
    return None
  else:
    return result