import os
import sys
import copy
import time
import json
import torch
import random
import numpy as np
from tqdm import tqdm

from agents import *
from utils.helper import *

AGENTS = {
    'NaiveDQN': NaiveDQN,
    'DQN': DQN,
    'DDQN': DDQN,
    'MaxminDQN': MaxminDQN
  }


class Experiment(object):
  '''
  Train the agent to play the game.
  '''
  def __init__(self, cfg):
    self.results = None
    self.cfg = copy.deepcopy(cfg)
    self.runs = cfg.runs
    self.config_idx = cfg.config_idx
    self.exp_name = cfg.exp_name
    if self.cfg.generate_random_seed:
      self.cfg.seed = random.randint(0, 2**32 - 2)
    
    self.time_str = get_time_str()
    self.cfg.tag = f'{self.exp_name}-{cfg.agent}-{self.config_idx}-{self.time_str}'
    self.cfg.log_dir = f'{cfg.log_dir}{self.cfg.tag}/'
    if not os.path.exists(self.cfg.log_dir): os.makedirs(self.cfg.log_dir)

    self.log_path = f'{self.cfg.log_dir}results.csv'
    self.model_path = f'{self.cfg.log_dir}model.pt'
    self.cfg_path = f'{self.cfg.log_dir}config.json'
    
    self.image_path = f'{cfg.image_dir}{self.cfg.tag}.png'
    
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
    for r in tqdm(range(self.runs)):
      self.agent = AGENTS[self.cfg.agent](self.cfg, r+1)
      self.set_random_seed(self.cfg.seed + r)
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

  def save_config(self):
    cfg_json = json.dumps(self.cfg.__dict__, indent=2)
    print(cfg_json, end='\n')
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()