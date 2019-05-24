import sys
import time
import random
import logging

import torch
import numpy as np
from tqdm import tqdm

from agents import *
from utils.helper import *

AGENTS = {
    'DQN': DQN
  }

  
class Experiment(object):
  # Train the agent to play the game.
  def __init__(self, cfg):
    self.agent = AGENTS[cfg.agent](cfg)
    self.runs = cfg.runs
    self.config_idx = cfg.config_idx
    self.exp_name = cfg.exp_name
    self.log_name = f'./logs/{self.exp_name}-{self.config_idx}.csv'
    self.image_name = f'./images/{self.exp_name}-{self.config_idx}.png'
    self.results = None
    self.set_random_seed(cfg)

  def set_random_seed(self, cfg):
    # Set all random seeds
    if cfg.generate_random_seed:
      cfg.seed = random.randint(0, 2**32 - 2)
    self.seed = cfg.seed
    random.seed(self.seed)
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available(): 
      torch.cuda.manual_seed_all(self.seed)
    # self.agent.set_random_seed(self.seed)
    # self.agent.env.set_random_seed(self.seed)

  def run(self):
    # Run the game for multiple times
    self.start_time = time.time()
    for r in tqdm(range(self.runs)):
      print(f'Run: {r+1}/{self.runs}')
      result = self.agent.run_episodes()
      if r == 0:
        self.results = result
      else:
        self.results = self.results.append(result, ignore_index=True)
      self.save_results()
    self.end_time = time.time()
    print(f'Memory usage: {rss_memory_usage():.2f} MB')
    print(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')

  def save_results(self):
    self.results.to_csv(self.log_name, index=False)