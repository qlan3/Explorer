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
from agents.BaseAgent import BaseAgent


class VanillaDQN(BaseAgent):
  '''
  Implementation of Vanilla DQN with only replay buffer (no target network)
  '''
  def __init__(self, cfg, run):
    super().__init__(cfg, run)
    self.agent_name = cfg.agent
    self.env = make_env(cfg.env)
    self.device = torch.device(cfg.device)
    self.batch_size = cfg.batch_size
    self.discount = cfg.discount
    self.exploration_steps = int(cfg.exploration['exploration_steps'])
    self.time_out_step = int(cfg.time_out_step)
    self.train_max_episodes = int(cfg.train_max_episodes)
    self.test_max_episodes = int(cfg.test_max_episodes)
    self.display_interval = cfg.display_interval
    self.gradient_clip = cfg.gradient_clip
    self.sgd_update_frequency = int(cfg.sgd_update_frequency)
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    self.rolling_score_window = cfg.rolling_score_window
    self.input_type = cfg.input_type

    if self.input_type == 'pixel':
      self.layer_dims = [cfg.feature_dim] + cfg.hidden_layers + [self.action_size]
      # Create Q value network
      feature_net = Conv2d_NN(in_channels=cfg.history_length, feature_dim=cfg.feature_dim)
      value_net = MLP(layer_dims=self.layer_dims)
      self.Q_net = NetworkGlue(feature_net, value_net).to(self.device)
    elif self.input_type == 'feature':
      self.layer_dims = [self.state_size] + cfg.hidden_layers + [self.action_size]
      self.Q_net = MLP(layer_dims=self.layer_dims,hidden_activation=nn.ReLU()).to(self.device)
    else:
      raise ValueError(f'{self.input_type} is not supported.')
    
    # Replay buffer
    self.replay_buffer = Replay(int(cfg.memory_size), cfg.batch_size, cfg.device)
    # Exploration strategy
    # self.exploration = LinearEpsilonGreedy(cfg.exploration)
    # self.exploration = EpsilonGreedy(cfg.exploration)
    self.exploration = ExponentialEpsilonGreedy(cfg.exploration)
    # Loss function
    self.loss = nn.MSELoss(reduction='mean')
    # Optimizer
    self.optimizer = optim.RMSprop(self.Q_net.parameters(), lr=cfg.lr)
    # self.optimizer = optim.Adam(self.Q_net.parameters(), lr=cfg.lr)

  def reset_game(self):
    # Reset the game before a new episode
    self.state = self.env.reset()
    self.next_state = None
    self.action = None
    self.reward = None
    self.done = False
    self.total_episode_reward = 0
    self.episode_step_count = 0

  def run_episodes(self, mode='Train', render=False):
    # Run for multiple episodes
    self.step_count = 0
    self.episode_count = 0
    result = []
    total_episode_reward_list = []

    if mode=='Train':
      max_episodes = self.train_max_episodes
    elif mode=='Test':
      max_episodes = self.test_max_episodes
    while self.episode_count < max_episodes:
      # Run for one episode
      self.run_episode(mode, render)
      # Save result
      total_episode_reward_list.append(self.total_episode_reward)
      rolling_score = np.mean(total_episode_reward_list[-1 * self.rolling_score_window:])
      result_dict = {'Agent': self.agent_name, 
                     'Episode': self.episode_count, 
                     'Step': self.step_count, 
                     'Return': self.total_episode_reward,
                     'Rolling Return': rolling_score}
      result.append(result_dict)
      # self.logger.add_scalars(f'[{mode}] Return',{f'run{self.run}': self.total_episode_reward}, self.episode_count)
      if self.episode_count % self.display_interval == 0:
        self.logger.info(f'[{mode}] Episode {self.episode_count}, Step {self.step_count}: Rolling Return({self.rolling_score_window})={rolling_score:.2f}, Return={self.total_episode_reward:.2f}')
    
    return pd.DataFrame(result)
  
  def run_episode(self, mode, render):
    # Run for one episode
    self.reset_game()
    while not (self.done or self.episode_step_count >= self.time_out_step):
      self.action = self.get_action(mode) # Take a step
      if render:
        self.env.render()
      self.next_state, self.reward, self.done, _ = self.env.step(self.action)
      if mode=='Train':
        self.save_experience()
        if self.time_to_learn(): self.learn() # Update policy
      self.state = self.next_state
      self.episode_step_count += 1
      self.step_count += 1
      self.total_episode_reward += self.reward
    self.episode_count += 1

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network and an epsilon greedy policy to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state, device=self.device)
    # Add a batch dimension (Batch, Channel, Height, Width)
    state = state.unsqueeze(0)
    self.Q_net.eval() # Set network in evaluation mode
    with torch.no_grad():
      q_values = self.Q_net(state)
    if mode=='Train':
      self.Q_net.train() # Set network back in training mode
      action = self.exploration.select_action(q_values, self.step_count)
    elif mode=='Test':
      action = torch.argmax(q_values).item() # During test, select action greedily
    return action

  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
    - The agent is not in exploration stage
    - There are enough experiences in replay buffer
    """
    if len(self.replay_buffer) > self.batch_size and \
      self.step_count > self.exploration_steps and \
      self.step_count % self.sgd_update_frequency == 0:
      return True
    else:
      return False

  def learn(self):
    states, actions, next_states, rewards, dones = self.replay_buffer.sample()
    
    self.logger.debug(f'states: {states.size()}')
    self.logger.debug(f'actions: {actions.size()}')
    self.logger.debug(f'next_states: {next_states.size()}')
    self.logger.debug(f'rewards: {rewards.size()}')
    self.logger.debug(f'dones: {dones.size()}')

    # Compute q target
    q_target = self.compute_q_target(next_states, rewards, dones)
    # Compute q
    q = self.comput_q(states, actions)
    
    self.logger.debug(f'q size: {q.size()}')
    self.logger.debug(f'q target size: {q_target.size()}')
    
    # Take an optimization step
    loss = self.loss(q, q_target)

    self.logger.debug(f'Step {self.step_count}: loss={loss.item()}')
    
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.Q_net.parameters(), self.gradient_clip)
    self.optimizer.step()

  def compute_q_target(self, next_states, rewards, dones):
    with torch.no_grad():
      q_target = self.Q_net(next_states).detach().max(1)[0]
      q_target = rewards + self.discount * q_target * (1 - dones)
    return q_target
  
  def comput_q(self, states, actions):
    # Convert actions to long so they can be used as indexes
    actions = actions.long()
    q = self.Q_net(states)
    q = q.gather(1, actions).squeeze()
    return q

  def save_experience(self):
    # Saves recent experience to replay buffer 
    experience = [self.state, self.action, self.next_state, self.reward, self.done]
    self.replay_buffer.add([experience])

  def get_action_size(self):
    if isinstance(self.env.action_space, Discrete):
      return self.env.action_space.n
    elif isinstance(self.env.action_space, Box):
      return self.env.action_space.shape[0]
    else:
      raise ValueError('Unknown action type.')
    
  def get_state_size(self):
    return int(np.prod(self.env.observation_space.shape))