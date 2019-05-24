import sys
import time
import json
import torch
import random
import logging
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
    self.results = None
    self.log_path = f'{cfg.log_dir}{self.exp_name}-{cfg.agent}-index={self.config_idx}.csv'
    self.image_path = f'{cfg.image_dir}{self.exp_name}-{cfg.agent}-index={self.config_idx}.png'
    self.model_path = f'{cfg.model_dir}{self.exp_name}-{cfg.agent}-index={self.config_idx}.pt'
    self.cfg_path = f'{cfg.log_dir}{self.exp_name}-{cfg.agent}-index={self.config_idx}.json'
    self.set_random_seed(cfg)
    self.save_config(cfg)

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
      self.save_model()
    self.end_time = time.time()
    print(f'Memory usage: {rss_memory_usage():.2f} MB')
    print(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')

  def save_results(self):
    self.results.to_csv(self.log_path, index=False)
  
  def save_model(self):
    torch.save(self.agent.Q_net.state_dict(), self.model_path)
  
  def load_model(self):
    self.agent.Q_net.load_state_dict(torch.load(self.model_path))

  def save_config(self, cfg):
    cfg_json = json.dumps(cfg.__dict__, indent=2)
    print(cfg_json, end='\n')
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()