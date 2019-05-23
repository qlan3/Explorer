import sys
import time
import random
import logging

import torch
import numpy as np

from utils.helper import *

from agents import *

AGENTS = {
    'DQN': DQN
  }

  
class Experiment(object):
  # Train the agent to play the game.
  def __init__(self, cfg):
    self.agent = AGENTS[cfg.agent](cfg)
    self.runs = cfg.runs
    self.set_random_seed(cfg)

  def set_random_seed(self, cfg):
    # Set all random seeds to reproduce results
    if cfg.generate_random_seed:
      self.seed = random.randint(0, 2**32 - 2)
    else:
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
    for r in range(self.runs):
      self.agent.run_episodes()
    self.end_time = time.time()
    print(f'Memory usage: {rss_memory_usage():5} MB')
    print(f'Time elapsed: {(self.end_time-self.start_time)/60:5.2} minutes')