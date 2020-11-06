from envs.env import *
from utils.helper import *
from agents.BaseAgent import *
from components.replay import *
from components.network import *
from components.normalizer import *


class A2C(BaseAgent):
  '''
  Implementation of Actor-Critic
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.cfg = cfg
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    self.env = make_env(cfg['env']['name'], max_episode_steps=int(cfg['max_episode_steps']))
    self.config_idx = cfg['config_idx']
    self.device = cfg['device']
    self.discount = cfg['discount']
    self.train_steps = int(cfg['train_steps'])
    self.test_per_episodes = int(cfg['test_per_episodes'])
    self.steps_per_epoch = cfg['steps_per_epoch']
    self.display_interval = cfg['display_interval']
    self.gradient_clip = cfg['gradient_clip']
    self.action_size = self.get_action_size()
    self.state_size = self.get_state_size()
    self.rolling_score_window = cfg['rolling_score_window']
    self.show_tb = cfg['show_tb']
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
    # Create policy network
    self.hidden_activation, self.output_activation = cfg['hidden_activation'], cfg['output_activation']
    self.network = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs']),
      'critic':  getattr(torch.optim, cfg['optimizer']['name'])(self.network.critic_params, **cfg['optimizer']['critic_kwargs'])
    }
    # Set storage
    self.storage = Storage(self.steps_per_epoch, keys=['reward', 'mask', 'v', 'log_prob', 'ret', 'adv'])

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers']
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env.game.state_shape()[2], feature_dim=layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    elif input_type == 'feature':
      layer_dims = [self.state_size] + self.cfg['hidden_layers']
      feature_net = nn.Identity()
    # Set actor network
    if self.action_type == 'DISCRETE':
      actor_net = MLPCategoricalActor(layer_dims=layer_dims+[self.action_size], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    elif self.action_type == 'CONTINUOUS':
      actor_net = MLPGaussianActor(layer_dims=layer_dims+[self.action_size], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    # Set critic network
    critic_net = MLPCritic(layer_dims=layer_dims+[1], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    # Set the model
    NN = ActorCriticNet(feature_net, actor_net, critic_net)
    return NN

  def reset_game(self):
    # Reset the game before a new episode
    self.state = self.state_normalizer(self.env.reset())
    self.next_state = None
    self.action = None
    self.log_prob = None
    self.reward = None
    self.done = False
    self.episode_return = 0

  def save_experience(self, prediction):
    if self.reward is not None:
      prediction['mask'] = to_tensor(1-self.done, self.device)
      prediction['reward'] = to_tensor(self.reward, self.device)
      # prediction['state'] = to_tensor(self.state, self.device)
      # prediction['next_state'] = to_tensor(self.next_state, self.device)
    self.storage.add(prediction)

  def run_steps(self, render=False):
    # Run for multiple episodes
    self.step_count = 0
    self.episode_count = 0
    self.result = {'Train': [], 'Test': []}
    self.episode_return_list = {'Train': [], 'Test': []}
    mode = 'Train'
    self.start_time = time.time()
    self.reset_game()
    while self.step_count < self.train_steps:
      if mode == 'Train' and self.episode_count % self.test_per_episodes == 0:
        mode = 'Test'
      else:
        mode = 'Train'
      # Set network back to training/evaluation mode
      self.set_net_mode(mode)
      # Run for one episode
      self.run_episode(mode, render)
    return pd.DataFrame(self.result['Train']), pd.DataFrame(self.result['Test'])

  def run_episode(self, mode, render):
    if mode == 'Train': 
      for _ in range(self.steps_per_epoch):
        prediction = self.get_action(mode)
        self.action = to_numpy(prediction['action'])
        if render:
          self.env.render()
        # Take a step
        self.next_state, self.reward, self.done, _ = self.env.step(self.action)
        self.next_state = self.state_normalizer(self.next_state)
        self.reward = self.reward_normalizer(self.reward)
        self.episode_return += self.reward  
        self.step_count += 1
        # Save experience
        self.save_experience(prediction)
        # Update state
        self.state = self.next_state
        # End of one episode
        if self.done:
          self.episode_count += 1
          self.save_episode_result(mode)
          self.reset_game()
      prediction = self.get_action(mode)
      self.save_experience(prediction)
      self.storage.placeholder()
      # Update policy
      self.learn()
      # Reset storage
      self.storage.empty()
    elif mode == 'Test':
      while not self.done:
        prediction = self.get_action(mode)
        self.action = to_numpy(prediction['action'])
        if render:
          self.env.render()
        # Take a step
        self.next_state, self.reward, self.done, _ = self.env.step(self.action)
        self.next_state = self.state_normalizer(self.next_state)
        self.reward = self.reward_normalizer(self.reward)
        self.episode_return += self.reward
        # Update state
        self.state = self.next_state
      # End of one episode
      self.episode_count += 1
      self.save_episode_result(mode)
      self.reset_game()
    
  def save_episode_result(self, mode):
    self.episode_return_list[mode].append(self.episode_return)
    rolling_score = np.mean(self.episode_return_list[mode][-1 * self.rolling_score_window[mode]:])
    result_dict = {'Env': self.env_name,
                   'Agent': self.agent_name,
                   'Episode': self.episode_count, 
                   'Step': self.step_count, 
                   'Return': self.episode_return,
                   'Average Return': rolling_score}
    self.result[mode].append(result_dict)
    if self.show_tb:
      self.logger.add_scalar(f'{mode}_Return', self.episode_return, self.step_count)
      self.logger.add_scalar(f'{mode}_Average_Return', rolling_score, self.step_count)
    if self.episode_count % self.display_interval == 0:
      speed = self.step_count / (time.time() - self.start_time)
      self.logger.info(f'<{self.config_idx}> [{mode}] Episode {self.episode_count}, Step {self.step_count}: Average Return({self.rolling_score_window[mode]})={rolling_score:.2f}, Return={self.episode_return:.2f}, Speed={speed:.2f}(steps/s)')

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    '''
    state = to_tensor(self.state, self.device)
    prediction = self.network(state)
    return prediction

  def learn(self):
    # Compute return and advantage
    adv = 0
    ret = self.storage.v[-1].detach()
    for i in reversed(range(self.steps_per_epoch)):
      ret = self.storage.reward[i] + self.discount * self.storage.mask[i] * ret
      if self.cfg['gae'] < 0:
        adv = ret - self.storage.v[i].detach()
      else:
        td_error = self.storage.reward[i] + self.discount * self.storage.mask[i] * self.storage.v[i+1] - self.storage.v[i]
        adv = self.discount * self.cfg['gae'] * self.storage.mask[i] * adv + td_error
      self.storage.adv[i] = adv.detach()
      self.storage.ret[i] = ret.detach()
    # Get training data
    entries = self.storage.get(['log_prob', 'v', 'ret', 'adv'], self.steps_per_epoch)
    # Compute losses
    actor_loss = -(entries.log_prob * entries.adv).mean()
    critic_loss = 0.5 * (entries.ret - entries.v).pow(2).mean()
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}')
    # Take an optimization step
    self.optimizer['actor'].zero_grad()
    self.optimizer['critic'].zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['actor'].step()
    self.optimizer['critic'].step()

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