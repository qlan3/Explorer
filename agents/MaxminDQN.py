import gym
import time
import torch
import numpy as np
import torch.optim as optim

from utils.helper import *
from agents.VanillaDQN import VanillaDQN


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer

  We can update all Q_nets for every update. However, this makes training really slow.
  Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    # Create k different: Q value network, Target Q value network and Optimizer
    self.Q_net = [None] * self.k
    self.Q_net_target = [None] * self.k
    self.optimizer = [None] * self.k
    for i in range(self.k):
      self.Q_net[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
      self.Q_net_target[i].eval()

  def learn(self):
    # Choose a Q_net to udpate
    self.update_Q_net_index = np.random.choice(list(range(self.k)))
    super().learn()
    # Update target network
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())

  def compute_q_target(self, next_states, rewards, dones):
    q_min = self.Q_net_target[0](next_states).clone().detach()
    for i in range(1, self.k):
      q = self.Q_net_target[i](next_states).detach()
      q_min = torch.min(q_min, q)
    q_next = q_min.max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_min = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_min = torch.min(q_min, q)
    q_min = to_numpy(q_min).flatten()
    return q_min