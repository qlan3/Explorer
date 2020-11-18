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
    self.cfg = copy.deepcopy(cfg)
    if torch.cuda.is_available() and 'cuda' in cfg['device']:
      self.device = cfg['device']
    else:
      self.cfg['device'] = 'cpu'
      self.device = 'cpu'
    self.config_idx = cfg['config_idx']
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    if self.cfg['generate_random_seed']:
      self.cfg['seed'] = np.random.randint(int(1e6))
    self.model_path = self.cfg['model_path']
    self.cfg_path = self.cfg['cfg_path']
    self.save_config()

  def run(self):
    '''
    Run the game for multiple times
    '''
    set_one_thread()
    self.start_time = time.time()
    set_random_seed(self.cfg['seed'])
    self.agent = getattr(agents, self.agent_name)(self.cfg)
    self.agent.env['Train'].seed(self.cfg['seed'])
    self.agent.env['Test'].seed(self.cfg['seed'])
    # Train && Test
    self.agent.run_steps(render=self.cfg['render'])
    # Save model
    # self.save_model()
    self.end_time = time.time()
    self.agent.logger.info(f'Memory usage: {rss_memory_usage():.2f} MB')
    self.agent.logger.info(f'Time elapsed: {(self.end_time-self.start_time)/60:.2f} minutes')
  
  def save_model(self):
    self.agent.save_model(self.model_path)
  
  def load_model(self):
    self.agent.load_model(self.model_path)

  def save_config(self):
    cfg_json = json.dumps(self.cfg, indent=2)
    f = open(self.cfg_path, 'w')
    f.write(cfg_json)
    f.close()