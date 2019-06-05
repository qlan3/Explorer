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
from agents.DQN import DQN


class MaxminDQN(DQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer
  '''
  def __init__(self, cfg, run):
    super().__init__(cfg, run)

  def compute_q_target(self, next_states, rewards, dones):
    with torch.no_grad():
      q1 = self.Q_net(next_states).detach()
      q2 = self.Q_net_target(next_states).detach()
      self.logger.debug('q1 & q2 size:', q1.size())
      
      q_min = torch.min(q1, q2)
      self.logger.debug('q min size:', q_min.size())
      
      best_actions = q_min.argmax(1).unsqueeze(1)
      self.logger.debug('best actions:', best_actions.size())
      
      q_target = q_min.gather(1, best_actions).squeeze()
      self.logger.debug('maxmin q target size 0:', q_target.size())
      
      q_target = rewards + self.discount * q_target * (1 - dones)
      self.logger.debug('maxin q target size 1:', q_target.size())
    
    return q_target