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
    self.env = {
      'Train': make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps']))
      'Test': make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps']))
    } 
    self.config_idx = cfg['config_idx']
    self.device = cfg['device']
    self.discount = cfg['discount']
    self.train_steps = int(cfg['train_steps'])
    self.test_per_episodes = int(cfg['test_per_episodes'])
    self.display_interval = cfg['display_interval']
    self.gradient_clip = cfg['gradient_clip']
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    self.rolling_score_window = cfg['rolling_score_window']
    self.show_tb = cfg['show_tb']
    self.log_path = {'Train': self.cfg['train_log_path'], 'Test': self.cfg['test_log_path']}
    # Set tensorboard
    if self.show_tb: self.logger.init_writer()
    # Set normalizers
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
    # Create Q value network
    self.hidden_activation, self.output_activation = cfg['hidden_activation'], 'None'
    self.Q_net = [None]
    self.Q_net[0] = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set optimizer
    self.optimizer = [None]
    self.optimizer[0] = getattr(torch.optim, cfg['optimizer']['name'])(self.Q_net[0].parameters(), **cfg['optimizer']['kwargs'])
    # Set replay buffer
    self.replay = getattr(components.replay, cfg['memory_type'])(cfg['memory_size'], self.cfg['batch_size'], self.device)
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
    # Set the index of Q_net to be udpated
    self.update_Q_net_index = 0
    # Set log dict
    for key in ['state', 'next_state', 'action', 'reward', 'done', 'episode_return', 'episode_step_count']:
      setattr(self, key, {'Train': None, 'Test': None})

  def createNN(self, input_type):
    if input_type == 'pixel':
      layer_dims = [cfg['feature_dim']] + cfg['hidden_layers'] + [self.action_size]
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env[mode].game.state_shape()[2], feature_dim=layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    elif input_type == 'feature':
      layer_dims = [self.state_size] + cfg['hidden_layers'] + [self.action_size]
      feature_net = nn.Identity()
    
    value_net = MLP(layer_dims=layer_dims, hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    NN = DQNNet(feature_net, value_net)
    return NN

  def reset_game(self, mode):
    # Reset the game before a new episode
    self.state[mode] = self.state_normalizer(self.env[mode].reset())
    self.next_state[mode] = None
    self.action[mode] = None
    self.reward[mode] = None
    self.done[mode] = False
    self.episode_return[mode] = 0
    self.episode_step_count[mode] = 0

  def run_steps(self, render=False):
    # Run for multiple episodes
    self.step_count = 0
    self.episode_count = 0
    self.result = {'Train': [], 'Test': []}
    self.episode_return_list = {'Train': [], 'Test': []}
    mode = 'Train'
    self.start_time = time.time()
    self.reset_game('Train')
    self.reset_game('Test')
    while self.step_count < self.train_steps:
      if mode == 'Train' and self.episode_count % self.test_per_episodes == 0:
        mode = 'Test'
      else:
        mode = 'Train'
      # Set Q network to training/evaluation mode
      self.set_net_mode(mode)
      # Run for one episode
      self.run_episode(mode, render)
    self.save_episode_result('Train')
    self.save_episode_result('Test')

  def run_episode(self, mode, render):
    while not self.done[mode]:
      self.action[mode] = self.get_action(mode) # Take a step
      if render:
        self.env[mode].render()
      # Take a step
      self.next_state[mode], self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
      self.next_state[mode] = self.state_normalizer(self.next_state[mode])
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_return[mode] += self.reward[mode]
      self.episode_step_count[mode] += 1
      if mode == 'Train':
        self.step_count += 1
        # Save experience
        self.save_experience()
        if self.time_to_learn():
          self.learn() # Update policy
      # Update state
      self.state[mode] = self.next_state[mode]
    # End of one episode
    self.save_episode_result(mode)
    if mode == 'Train':
      self.episode_count += 1
    # Reset environment
    self.reset_game(mode)

  def save_episode_result(self, mode):
    self.episode_return_list[mode].append(self.episode_return[mode])
    rolling_score = np.mean(self.episode_return_list[mode][-1 * self.rolling_score_window[mode]:])
    result_dict = {'Env': self.env_name,
                   'Agent': self.agent_name,
                   'Episode': self.episode_count, 
                   'Step': self.step_count, 
                   'Return': self.episode_return[mode],
                   'Average Return': rolling_score}
    self.result[mode].append(result_dict)
    if self.show_tb:
      self.logger.add_scalar(f'{mode}_Return', self.episode_return[mode], self.step_count)
      self.logger.add_scalar(f'{mode}_Average_Return', rolling_score, self.step_count)
    if mode == 'Test' or self.episode_count % self.display_interval == 0 or self.step_count >= self.train_steps:
      # Save result to files
      result = pd.DataFrame(self.result[mode])
      result['Env'] = result['Env'].astype('category')
      result['Agent'] = result['Agent'].astype('category')
      result.to_feather(self.log_path[mode])
      # Show log
      speed = self.step_count / (time.time() - self.start_time)
      self.logger.info(f'<{self.config_idx}> [{mode}] Episode {self.episode_count}, Step {self.step_count}: Average Return({self.rolling_score_window[mode]})={rolling_score:.2f}, Return={self.episode_return[mode]:.2f}, Speed={speed:.2f}(steps/s)')

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network and an epsilon greedy policy to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
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
    if self.step_count > self.exploration_steps and self.step_count % self.cfg['network_update_frequency'] == 0:
      return True
    else:
      return False

  def learn(self):
    mode = 'Train'
    states, actions, next_states, rewards, dones = self.replay.sample()
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
    mode = 'Train'
    # Saves recent experience to replay buffer
    experience = [self.state[mode], self.action[mode], self.next_state[mode], self.reward[mode], self.done[mode]]
    self.replay.add([experience])

  def get_action_size(self):
    mode = 'Train'
    assert isinstance(self.env[mode].action_space, Discrete), f'{self.agent_name} only supports discrete action space!'
    return self.env[mode].action_space.n
    
  def get_state_size(self):
    mode = 'Train'
    return int(np.prod(self.env[mode].observation_space.shape))

  def set_net_mode(self, mode):
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