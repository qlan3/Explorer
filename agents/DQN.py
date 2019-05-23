import gym
import torch
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


class DQN(BaseAgent):
  def __init__(self, cfg):
    super().__init__(cfg)
    self.env = make_env(cfg.env)
    self.device = torch.device(cfg.device)
    self.batch_size = cfg.batch_size
    self.discount = cfg.discount
    self.exploration_steps = int(cfg.exploration_steps)
    self.time_out_step = int(cfg.time_out_step)
    self.max_episodes = int(cfg.max_episodes)
    self.max_steps = int(cfg.max_steps)
    self.gradient_clip = cfg.gradient_clip
    self.target_network_update_freq = int(cfg.target_network_update_freq)
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    if len(self.env.observation_space.shape) == 3:
      self.input_type = 'pixel'
      layer_dims = [cfg.feature_dim] + cfg.hidden_layers + [self.action_size]
    else:
      self.input_type = 'feature'
      layer_dims = [self.state_size] + cfg.hidden_layers + [self.action_size]
      # print(layer_dims)
    # Reset counter
    self.step_count = 0
    self.episode_count = 0
    self.total_episode_reward_list = []
    
    if self.input_type == 'pixel':
      # Create Q value network
      feature_net = Conv2d_NN(in_channels=cfg.history_length, feature_dim=cfg.feature_dim)
      value_net = MLP(layer_dims=layer_dims)
      self.Q_net = NetworkGlue(feature_net, value_net)
      # Create target Q value network
      feature_net_target = Conv2d_NN(in_channels=cfg.history_length, feature_dim=cfg.feature_dim)
      value_net_target = MLP(layer_dims=layer_dims)
      self.Q_net_target = NetworkGlue(feature_net_target, value_net_target)
    elif self.input_type == 'feature':
      self.Q_net = MLP(layer_dims=layer_dims,hidden_activation=nn.Sigmoid())
      self.Q_net_target = MLP(layer_dims=layer_dims,hidden_activation=nn.Sigmoid())
    else:
      raise ValueError(f'{self.input_type} is not supported.')
    
    self.Q_net_target.load_state_dict(self.Q_net.state_dict())
    # Replay buffer
    self.replay_buffer = Replay(int(cfg.memory_size), cfg.batch_size, cfg.device)
    # Exploration strategy
    self.exploration = LinearEpsilonGreedy(cfg.exploration)
    # Loss function
    self.loss = nn.MSELoss(reduction='mean')
    # Optimizer
    ''' self.optimizer = optim.RMSprop(
      self.Q_net.parameters(), lr=cfg.lr, alpha=0.95, eps=0.01, centered=True) '''
    self.optimizer = optim.Adam(self.Q_net.parameters(), lr=cfg.lr)
  
  def reset_game(self):
    # Reset the game before a new episode
    self.state = self.env.reset()
    self.next_state = None
    self.action = None
    self.reward = None
    self.done = False
    self.total_episode_reward = 0
    self.episode_step_count = 0
  
  def run_episodes(self):
    self.reset_game()
    while (self.step_count < self.max_steps) and (self.episode_count < self.max_episodes):
      # Take a step
      self.action = self.get_action()
      # self.env.render()
      self.next_state, self.reward, self.done, _ = self.env.step(self.action)
      # Save experience
      self.save_experience()
      # Update
      if self.time_to_learn():
        self.learn()
      self.state = self.next_state
      self.episode_step_count += 1
      self.total_episode_reward += self.reward
      # If done, reset the game
      if self.done or self.episode_step_count >= self.time_out_step:
        self.total_episode_reward_list.append(self.total_episode_reward)
        self.episode_count += 1
        self.step_count += self.episode_step_count
        print(f'Episode {self.episode_count}, Step {self.step_count}: return={self.total_episode_reward}')
        self.reset_game()

  def get_action(self):
    # Uses the local Q network and an epsilon greedy policy to pick an action
    # PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    # a "fake" dimension to make it a mini-batch rather than a single observation
    state = self.state
    state = to_tensor(state, device=self.device)
    # Add a batch dimension (Batch, Channel, Height, Width)
    state = state.unsqueeze(0)
    
    self.Q_net.eval() # Set network in evaluation mode
    with torch.no_grad():
      q_values = self.Q_net(state)
    self.Q_net.train() # Set network back in training mode
    
    action = self.exploration.select_action(q_values)
    return action

  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
    - The agent is not in exploration stage
    - There are enough experiences in replay buffer
    """
    if len(self.replay_buffer) > self.batch_size and self.step_count > self.exploration_steps:
      return True
    else:
      return False

  def learn(self):
    states, actions, next_states, rewards, dones = self.replay_buffer.sample()
    '''
    print('states:', states.size())
    print('actions:', actions.size())
    print('next_states:', next_states.size())
    print('rewards:', rewards.size())
    print('dones:', dones.size())
    '''
    # Convert actions to long so they can be used as indexes
    actions = actions.long()
    # Compute q target
    with torch.no_grad():
      q_target = self.Q_net_target(next_states).detach().max(1)[0]
    q_target = rewards + self.discount * q_target * (1 - dones)
    # Compute q
    q = self.Q_net(states).gather(1, actions).squeeze()
    # Take an optimization step
    '''
    print('q size:', q.size())
    print('q target size:', q_target.size())
    '''
    loss = self.loss(q, q_target)
    # print(f'Step {self.step_count}: loss={loss.item()}')
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.Q_net.parameters(), self.gradient_clip)
    self.optimizer.step()
    # Update target network
    if self.step_count % self.target_network_update_freq == 0:
      self.Q_net_target.load_state_dict(self.Q_net.state_dict())

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