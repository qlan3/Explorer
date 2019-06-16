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


class DQN(VanillaDQN):
  '''
  Implementation of DQN with target network and replay buffer
  '''
  def __init__(self, cfg, run):
    super().__init__(cfg, run)
    self.target_network_update_freq = cfg.target_network_update_freq
    # Create target Q value network
    if self.input_type == 'pixel':
      feature_net_target = Conv2d_NN(in_channels=cfg.history_length, feature_dim=cfg.feature_dim)
      value_net_target = MLP(layer_dims=self.layer_dims)
      self.Q_net_target = NetworkGlue(feature_net_target, value_net_target).to(self.device)
    elif self.input_type == 'feature':
      self.Q_net_target = MLP(layer_dims=self.layer_dims,hidden_activation=nn.ReLU()).to(self.device)
    else:
      raise ValueError(f'{self.input_type} is not supported.')
    # Load target Q value network
    self.Q_net_target.load_state_dict(self.Q_net.state_dict())

  def learn(self):
    super().learn()
    # Update target network
    if self.step_count % self.target_network_update_freq == 0:
      self.Q_net_target.load_state_dict(self.Q_net.state_dict())

  def compute_q_target(self, next_states, rewards, dones):
    with torch.no_grad():
      q_target = self.Q_net_target(next_states).detach().max(1)[0]
      q_target = rewards + self.discount * q_target * (1 - dones)
    return q_target