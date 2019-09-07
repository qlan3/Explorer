import gym
import time
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from envs.env import *
from utils.helper import *
from components.network import *
from components.normalizer import *
import components.replay
import components.exploration
from agents.VanillaDQN import VanillaDQN


class MaxminDQN_v1(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.target_network_update_freqency = cfg['target_network_update_freqency']
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    # Create target Q value network
    self.Q_net_target = [None] * self.k
    for i in range(self.k):
      self.Q_net_target[i] = self.creatNN(cfg['env']['input_type']).to(self.device)
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net.state_dict())
      self.Q_net_target[i].eval()
    self.update_target_net_index = 0

  def learn(self):
    super().learn()
    # Update target network
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      self.Q_net_target[self.update_target_net_index].load_state_dict(self.Q_net.state_dict())
      self.update_target_net_index = (self.update_target_net_index + 1) % self.k
 
  def compute_q_target(self, next_states, rewards, dones):
    q_min = self.Q_net(next_states).clone().detach()
    for i in range(self.k):
      q = self.Q_net_target[i](next_states).detach()
      q_min = torch.min(q_min, q)
    q_next = q_min.max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.target_network_update_freqency = cfg['target_network_update_freqency']
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    '''
    # Create k different
    - Q value network
    - Target Q value network
    - Optimizer
    '''
    self.Q_net = [None] * self.k
    self.Q_net_target = [None] * self.k
    self.optimizer = [None] * self.k
    for i in range(self.k):
      self.Q_net[i] = self.creatNN(cfg['env']['input_type']).to(self.device)
      self.Q_net_target[i] = self.creatNN(cfg['env']['input_type']).to(self.device)
      self.optimizer[i] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[i].parameters(), **cfg['optimizer']['kwargs'])
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
      self.Q_net_target[i].eval()

  def run_steps(self, render=False):
    # Run for multiple episodes
    self.step_count = 0
    self.episode_count = 0
    result = {'Train': [], 'Test': []}
    rolling_score = {'Train': 0.0, 'Test': 0.0}
    total_episode_reward_list = {'Train': [], 'Test': []}
    mode = 'Train'
    while self.step_count < self.train_steps:
      if mode == 'Train' and self.episode_count % self.test_per_episodes == 0:
        mode = 'Test'
        for i in range(self.k):
          self.Q_net[i].eval() # Set network to evaluation mode
      else:
        mode = 'Train'
        for i in range(self.k):
          self.Q_net[i].train() # Set network back to training mode
      # Run for one episode
      start_time = time.time()
      start_step_count = self.step_count
      self.run_episode(mode, render)
      end_time = time.time()
      end_step_count = self.step_count + 1
      speed = (end_step_count - start_step_count) / (end_time - start_time)
      # Save result
      total_episode_reward_list[mode].append(self.total_episode_reward)
      rolling_score[mode] = np.mean(total_episode_reward_list[mode][-1 * self.rolling_score_window[mode]:])
      result_dict = {'Env': self.env_name,
                     'Agent': self.agent_name,
                     'Episode': self.episode_count, 
                     'Step': self.step_count, 
                     'Return': self.total_episode_reward,
                     'Average Return': rolling_score[mode]}
      result[mode].append(result_dict)
      if self.show_tb:
        self.logger.add_scalar(f'{mode}_Return', self.total_episode_reward, self.step_count)
        self.logger.add_scalar(f'{mode}_Average_Return', rolling_score[mode], self.step_count)
      if self.episode_count % self.display_interval == 0:
        self.logger.info(f'<{self.config_idx}> [{mode}] Episode {self.episode_count}, Step {self.step_count}: Average Return({self.rolling_score_window[mode]})={rolling_score[mode]:.2f}, Return={self.total_episode_reward:.2f}, Speed={speed:.2f}(steps/s)')

    return pd.DataFrame(result['Train']), pd.DataFrame(result['Test'])

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network and an epsilon greedy policy to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state, device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    
    # q_values = self.Q_net(state)
    # q_values = to_numpy(q_values).flatten()
    q_min = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_min = torch.min(q_min, q)
    q_values = to_numpy(q_min).flatten()

    if mode == 'Train':
      action = self.exploration.select_action(q_values, self.step_count)
    elif mode == 'Test':
      action = np.argmax(q_values) # During test, select best action
    return action

  def learn(self):
    i = np.random.choice(list(range(self.k)))
    states, actions, next_states, rewards, dones = self.replay_buffer.sample()
    # Compute q target
    q_target = self.compute_q_target(next_states, rewards, dones)
    # Compute q
    q = self.comput_q(states, actions, i)
    # Take an optimization step
    loss = self.loss(q, q_target)
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: loss={loss.item()}')
    self.optimizer[i].zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
    self.optimizer[i].step()
    
    # Update target network
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
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
  
  def comput_q(self, states, actions, i):
    # Convert actions to long so they can be used as indexes
    actions = actions.long()
    q = self.Q_net[i](states).gather(1, actions).squeeze()
    return q