import os
import sys
import copy
import time
import json
import torch
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
    self.game = cfg.game
    if self.cfg.generate_random_seed:
      self.cfg.seed = np.random.randint(int(1e6))
    self.train_log_path = self.cfg.train_log_path
    self.test_log_path = self.cfg.test_log_path
    self.model_path = self.cfg.model_path
    self.cfg_path = self.cfg.cfg_path
    self.save_config()

  def run(self):
    '''
    Run the game for multiple times
    '''
    set_one_thread()
    self.start_time = time.time()
    set_random_seed(self.cfg.seed)
    self.agent = getattr(agents, self.cfg.agent)(self.cfg)
    self.agent.env.seed(self.cfg.seed)
    # Train
    self.train_result = self.agent.run_episodes(mode='Train', render=False)
    self.save_results(mode='Train')
    # Test
    self.test_result = self.agent.run_episodes(mode='Test', render=False)
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
    # print(cfg_json, end='\n')
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()