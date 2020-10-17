import gym
import time
import copy
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
from agents.BaseAgent import BaseAgent


class REINFORCE(BaseAgent):
  '''
  Implementation of REINFORCE
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.cfg = cfg
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    self.max_episode_steps = int(cfg['env']['max_episode_steps'])
    self.env = make_env(cfg['env']['name'], max_episode_steps=self.max_episode_steps)
    self.config_idx = cfg['config_idx']
    self.device = cfg['device']
    self.discount = cfg['discount']
    self.train_steps = int(cfg['env']['train_steps'])
    self.test_per_episodes = int(cfg['env']['test_per_episodes'])
    self.display_interval = cfg['display_interval']
    self.gradient_clip = cfg['gradient_clip']
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    self.rolling_score_window = cfg['rolling_score_window']
    if 'MinAtar' in self.env_name:
      self.history_length = self.env.game.state_shape()[2]
    else:
      self.history_length = cfg['history_length']
    self.sgd_update_frequency = cfg['sgd_update_frequency']
    self.show_tb = cfg['show_tb']
    # Set tensorboard
    if self.show_tb: self.logger.init_writer()
    
    if cfg['env']['input_type'] == 'pixel':
      if 'MinAtar' in self.env_name:
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
      else:
        self.state_normalizer = ImageNormalizer()
        self.reward_normalizer = SignNormalizer()
    elif cfg['env']['input_type'] == 'feature':
      self.state_normalizer = RescaleNormalizer()
      self.reward_normalizer = RescaleNormalizer()
    else:
      raise ValueError(f"{cfg['env']['input_type']} is not supported.")
    self.hidden_activation, self.output_activation = cfg['hidden_activation'], cfg['output_activation']

    # Create policy network
    self.network = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set optimizer
    self.optimizer = getattr(torch.optim, cfg['optimizer']['name'])(self.network.parameters(), **cfg['optimizer']['kwargs'])

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      self.layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers']
      if 'MinAtar' in self.env_name:
        feature_net_head = Conv2d_MinAtar(in_channels=self.history_length, feature_dim=self.layer_dims[0])
      else:
        feature_net_head = Conv2d_Atari(in_channels=self.history_length, feature_dim=self.layer_dims[0])
      feature_net_body = MLP(layer_dims=self.layer_dims, hidden_activation=self.hidden_activation, output_activation=self.hidden_activation)
      feature_net = NetworkGlue(feature_net_head, feature_net_body)
    elif input_type == 'feature':
      self.layer_dims = [self.state_size] + self.cfg['hidden_layers']
      feature_net = MLP(layer_dims=self.layer_dims, hidden_activation=self.hidden_activation, output_activation=self.hidden_activation)
    # Set actor network and model
    if self.action_type == 'DISCRETE':
      actor_net = MLP(layer_dims=[self.layer_dims[-1], self.action_size], hidden_activation=self.hidden_activation, output_activation='Softmax-1')
      NN = CategoricalREINFORCENet(feature_net, actor_net)
    elif self.action_type == 'CONTINUOUS':
      actor_net = MLP(layer_dims=[self.layer_dims[-1], self.action_size], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
      NN = GaussianREINFORCENet(feature_net, actor_net, self.action_size)
      
    return NN

  def reset_game(self):
    # Reset the game before a new episode
    self.state = self.state_normalizer(self.env.reset())
    self.next_state = None
    self.action = None
    self.log_prob = 0.0
    self.reward = None
    self.done = False
    self.total_episode_reward = 0
    self.episode = {
      'step_count': 0,
      'reward': [],
      'done': [],
      'log_prob': []
    }

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
      else:
        mode = 'Train'
      self.set_net_mode(mode) # Set network back to training/evaluation mode
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
        print(f'<{self.config_idx}> [{mode}] Episode {self.episode_count}, Step {self.step_count}: Average Return({self.rolling_score_window[mode]})={rolling_score[mode]:.2f}, Return={self.total_episode_reward:.2f}, Speed={speed:.2f}(steps/s)')

    return pd.DataFrame(result['Train']), pd.DataFrame(result['Test'])

  def run_episode(self, mode, render):
    # Run for one episode
    self.reset_game()
    while not self.done:
      prediction = self.get_action(mode)
      self.action = prediction['action']
      if render:
        self.env.render()
      self.next_state, self.reward, self.done, _ = self.env.step(self.action) # Take a step
      self.next_state = self.state_normalizer(self.next_state)
      self.reward = self.reward_normalizer(self.reward)
      if mode == 'Train':
        self.save_experience(prediction)
        if self.time_to_learn():
          self.learn() # Update policy
        self.episode['step_count'] += 1
        self.step_count += 1
      self.total_episode_reward += self.reward
      self.state = self.next_state
    if mode == 'Train':
      self.episode_count += 1

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state, device=self.device)
    # Add a batch dimension (Batch, Channel, Height, Width)
    state = state.unsqueeze(0)
    prediction = self.network(state)
    action = to_numpy(prediction['action'])
    if action.size == 1 and 'CartPole' in self.env_name:
      action = action[0]
    prediction['action'] = action
    return prediction
    
  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
      For REINFORCE, only learn at the end of an episode.
    """
    return self.done

  def learn(self):
    # Calculate the cumulative discounted return for an episode
    discounted_returns = to_tensor(self.episode['reward'], device=self.device)
    dones = to_tensor(self.episode['done'], device=self.device)
    for i in range(len(discounted_returns)-2, -1, -1):
      discounted_returns[i] = discounted_returns[i] + (1-dones[i]) * self.discount * discounted_returns[i+1]
    # Compute loss
    loss = []
    for log_prob, discounted_return in zip(self.episode['log_prob'], discounted_returns):
      loss.append(-log_prob * discounted_return)
    loss = torch.cat(loss).sum()

    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: loss={loss.item()}')
    # Take an optimization step
    self.optimizer.zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
    self.optimizer.step()

  def save_experience(self, prediction):
    # Save reward and log_prob for loss computation
    self.episode['reward'].append(self.reward)
    self.episode['done'].append(self.done)
    self.episode['log_prob'].append(prediction['log_prob'])

  def get_action_size(self):
    if isinstance(self.env.action_space, Discrete):
      self.action_type = 'DISCRETE'
      return self.env.action_space.n
    elif isinstance(self.env.action_space, Box):
      self.action_type = 'CONTINUOUS'
      return self.env.action_space.shape[0]
    else:
      raise ValueError('Unknown action type.')
    
  def get_state_size(self):
    return int(np.prod(self.env.observation_space.shape))

  def set_net_mode(self, mode):
    if mode == 'Test':
      self.network.eval() # Set network to evaluation mode
    elif mode == 'Train':
      self.network.train() # Set network back to training mode

  def save_model(self, model_path):
    torch.save(self.network.state_dict(), model_path)
  
  def load_model(self, model_path):
    self.network.load_state_dict(torch.load(model_path))
    self.network = self.network.to(self.device)