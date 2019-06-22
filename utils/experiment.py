import os
import sys
import copy
import time
import json
import torch
import random
import numpy as np
import pandas as pd

import agents
from utils.helper import *


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.train_result = None
    self.test_result = None
    self.cfg = copy.deepcopy(cfg)
    self.config_idx = cfg.config_idx
    self.exp_name = cfg.exp_name
    if self.cfg.generate_random_seed:
      self.cfg.seed = random.randint(0, 2**32 - 2)
    self.cfg.logs_dir = f'{cfg.logs_dir}{self.config_idx}/'
    if not os.path.exists(self.cfg.logs_dir): os.makedirs(self.cfg.logs_dir)
    self.train_log_path = self.cfg.logs_dir + 'result-Train.csv'
    self.test_log_path = self.cfg.logs_dir + 'result-Test.csv'
    self.model_path = self.cfg.logs_dir + 'model.pt'
    self.cfg_path = self.cfg.logs_dir + 'config.json'
    self.save_config()

  def set_random_seed(self, seed):
    '''
    Set all random seeds
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    self.agent.env.seed(seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): 
      torch.cuda.manual_seed_all(seed)
    # self.agent.set_random_seed(seed)

  def run(self):
    '''
    Run the game for multiple times
    '''
    set_one_thread()
    self.start_time = time.time()
    self.agent = getattr(agents, self.cfg.agent)(self.cfg)
    self.set_random_seed(self.cfg.seed)
    # Train
    self.train_result = self.agent.run_episodes(mode='Train', render=False)
    self.save_results(mode='Train')
    # Test
    test_result = self.agent.run_episodes(mode='Test')
    test_result = {'Agent': self.agent.agent_name,
                   'Average Return': test_result['Return'].mean(),
                   'Average Rolling Return': test_result['Rolling Return'].mean()}
    self.test_result = pd.DataFrame([test_result])
    self.save_results(mode='Test')
    # Save model
    self.save_model()
    self.end_time = time.time()
    print(f'Memory usage: {rss_memory_usage():.2f} MB')
    print(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')

  def save_results(self, mode):
    if mode == 'Train':
      self.train_result.to_csv(self.train_log_path, index=False)
    elif mode == 'Test':
      self.test_result.to_csv(self.test_log_path, index=False)
  
  def save_model(self):
    torch.save(self.agent.Q_net.state_dict(), self.model_path)
  
  def load_model(self):
    self.agent.Q_net.load_state_dict(torch.load(self.model_path))

  def save_config(self):
    cfg_json = json.dumps(self.cfg.__dict__, indent=2)
    print(cfg_json, end='\n')
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()