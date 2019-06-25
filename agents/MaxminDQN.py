import gym
import torch
import pandas as pd
from torch import nn
import torch.optim as optim
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from envs.env import *
from utils.helper import *
from components.replay import *
from components.network import *
from components.exploration import *
from agents.VanillaDQN import VanillaDQN


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.target_network_update_freqency = cfg.target_network_update_freqency
    self.k = cfg.target_networks_num # number of target networks
    # Create target Q value network
    self.Q_net_target = [None] * self.k
    for i in range(self.k):
      self.Q_net_target[i] = self.creatNN(cfg.input_type).to(self.device)
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net.state_dict())
    self.update_target_net_index = 0

  def learn(self):
    super().learn()
    # Update target network
    if (self.step_count / self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      self.Q_net_target[self.update_target_net_index].load_state_dict(self.Q_net.state_dict())
      self.update_target_net_index = (self.update_target_net_index+1) % self.k
 

  def compute_q_target(self, next_states, rewards, dones):
    with torch.no_grad():
      q_min = self.Q_net(next_states).detach()
      # self.logger.debug('q min size 0:', q_min.size())
      
      for i in range(self.k):
        q = self.Q_net_target[i](next_states).detach()
        q_min = torch.min(q_min, q)
      
      # self.logger.debug('q min size 1:', q_min.size())
      
      best_actions = q_min.argmax(1).unsqueeze(1)
      # self.logger.debug('best actions:', best_actions.size())
      
      q_target = q_min.gather(1, best_actions).squeeze()
      # self.logger.debug('maxmin q target size 0:', q_target.size())
      
      q_target = rewards + self.discount * q_target * (1 - dones)
      # self.logger.debug('maxin q target size 1:', q_target.size())
    
    return q_target