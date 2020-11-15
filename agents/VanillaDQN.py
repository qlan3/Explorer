from envs.env import *
from utils.helper import *
from components.network import *
from components.normalizer import *
import components.replay
import components.exploration
from agents.BaseAgent import BaseAgent


class VanillaDQN(BaseAgent):
  '''
  Implementation of Vanilla DQN with only replay buffer (no target network)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    self.env = make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps']))
    self.config_idx = cfg['config_idx']
    self.device = cfg['device']
    self.batch_size = cfg['batch_size']
    self.discount = cfg['discount']
    self.train_steps = int(cfg['train_steps'])
    self.test_per_episodes = int(cfg['test_per_episodes'])
    self.display_interval = cfg['display_interval']
    self.gradient_clip = cfg['gradient_clip']
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    self.rolling_score_window = cfg['rolling_score_window']
    self.sgd_update_frequency = cfg['sgd_update_frequency']
    self.show_tb = cfg['show_tb']
    
    if cfg['env']['input_type'] == 'pixel':
      self.layer_dims = [cfg['feature_dim']] + cfg['hidden_layers'] + [self.action_size]
      if 'MinAtar' in self.env_name:
        self.state_normalizer = RescaleNormalizer()
        self.reward_normalizer = RescaleNormalizer()
      else:
        self.state_normalizer = ImageNormalizer()
        self.reward_normalizer = SignNormalizer()
    elif cfg['env']['input_type'] == 'feature':
      self.layer_dims = [self.state_size] + cfg['hidden_layers'] + [self.action_size]
      self.state_normalizer = RescaleNormalizer()
      self.reward_normalizer = RescaleNormalizer()
    else:
      raise ValueError(f"{cfg['env']['input_type']} is not supported.")
    self.hidden_activation, self.output_activation = cfg['hidden_activation'], 'None'
    
    # Create Q value network
    self.Q_net = [None]
    self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set optimizer
    self.optimizer = [None]
    self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(), **cfg['optimizer']['kwargs'])
    # Set replay buffer
    self.replay_buffer = getattr(components.replay, cfg['memory_type'])(cfg['memory_size'], self.batch_size, self.device)
    # Set exploration strategy
    epsilon = {
      'steps': float(cfg['epsilon_steps']),
      'start': cfg['epsilon_start'],
      'end': cfg['epsilon_end'],
      'decay': cfg['epsilon_decay']
      }
    self.exploration_steps = cfg['exploration_steps']
    # Set exploration strategy
    self.exploration = getattr(components.exploration, cfg['exploration_type'])(cfg['exploration_steps'], epsilon)
    # Set loss function
    self.loss = getattr(torch.nn, cfg['loss'])(reduction='mean')
    # Set tensorboard
    if self.show_tb: self.logger.init_writer()
    # Set the index of Q_net to be udpated
    self.update_Q_net_index = 0

  def createNN(self, input_type):
    if input_type == 'pixel':
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env.game.state_shape()[2], feature_dim=self.layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=self.layer_dims[0])
    elif input_type == 'feature':
      feature_net = nn.Identity()
    
    value_net = MLP(layer_dims=self.layer_dims, hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    NN = DQNNet(feature_net, value_net)
    return NN

  def reset_game(self):
    # Reset the game before a new episode
    self.state = self.state_normalizer(self.env.reset())
    self.next_state = None
    self.action = None
    self.reward = None
    self.done = False
    self.total_episode_reward = 0
    self.episode_step_count = 0

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
      self.set_Q_net_mode(mode) # Set Q network back to training/evaluation mode
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

  def run_episode(self, mode, render):
    # Run for one episode
    self.reset_game()
    while not self.done:
      self.action = self.get_action(mode) # Take a step
      if render:
        self.env.render()
      self.next_state, self.reward, self.done, _ = self.env.step(self.action)
      self.next_state = self.state_normalizer(self.next_state)
      self.reward = self.reward_normalizer(self.reward)
      if mode == 'Train':
        self.save_experience()
        if self.time_to_learn():
          self.learn() # Update policy
        self.episode_step_count += 1
        self.step_count += 1
      self.total_episode_reward += self.reward
      self.state = self.next_state
    if mode == 'Train':
      self.episode_count += 1

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network and an epsilon greedy policy to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state, device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    q_values = self.get_action_selection_q_values(state)
    if mode == 'Train':
      action = self.exploration.select_action(q_values, self.step_count)
    elif mode == 'Test':
      action = np.argmax(q_values) # During test, select best action
    return action

  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
    - The agent is not on exploration stage
    - There are enough experiences in replay buffer
    """
    if self.step_count > self.exploration_steps and self.step_count % self.sgd_update_frequency == 0:
      return True
    else:
      return False

  def learn(self):
    states, actions, next_states, rewards, dones = self.replay_buffer.sample()
    # Compute q target
    q_target = self.compute_q_target(next_states, rewards, dones)
    # Compute q
    q = self.comput_q(states, actions)
    # Take an optimization step
    loss = self.loss(q, q_target)
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: loss={loss.item()}')
    self.optimizer[self.update_Q_net_index].zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.Q_net[self.update_Q_net_index].parameters(), self.gradient_clip)
    self.optimizer[self.update_Q_net_index].step()

  def compute_q_target(self, next_states, rewards, dones):
    q_next = self.Q_net[0](next_states).detach().max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target
  
  def comput_q(self, states, actions):
    # Convert actions to long so they can be used as indexes
    actions = actions.long()
    q = self.Q_net[self.update_Q_net_index](states).gather(1, actions).squeeze()
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

  def set_Q_net_mode(self, mode):
    if mode == 'Test':
      for i in range(len(self.Q_net)):
        self.Q_net[i].eval() # Set Q network to evaluation mode
    elif mode == 'Train':
      for i in range(len(self.Q_net)):
        self.Q_net[i].train() # Set Q network back to training mode
  
  def get_action_selection_q_values(self, state):
    q_values = self.Q_net[0](state)
    q_values = to_numpy(q_values).flatten()
    return q_values

  def save_model(self, model_path):
    state_dicts = {} 
    for i in range(len(self.Q_net)):
      state_dicts[i] = self.Q_net[i].state_dict()
    torch.save(state_dicts, model_path)
  
  def load_model(self, model_path):
    state_dicts = torch.load(model_path)
    for i in range(len(self.Q_net)):
      self.Q_net[i].load_state_dict(state_dicts[i])
      self.Q_net[i] = self.Q_net[i].to(self.device)