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


class DDQN(DQN):
  '''
  Implementation of Double DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def compute_q_target(self, next_states, rewards, dones):
    best_actions = self.Q_net(next_states).detach().argmax(1).unsqueeze(1)
    q_next = self.Q_net_target(next_states).detach().gather(1, best_actions).squeeze()
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target