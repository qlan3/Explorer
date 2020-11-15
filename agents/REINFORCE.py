from envs.env import *
from utils.helper import *
from agents.BaseAgent import *
from components.replay import *
from components.network import *
from components.normalizer import *


class REINFORCE(BaseAgent):
  '''
  Implementation of REINFORCE
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.cfg = cfg
    self.env_name = cfg['env']['name']
    self.agent_name = cfg['agent']['name']
    self.env = make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps']))
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
    # Create policy network
    self.hidden_activation, self.output_activation = cfg['hidden_activation'], cfg['output_activation']
    self.network = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs'])
    }
    # Set replay buffer
    self.replay = InfiniteReplay(keys=['reward', 'mask', 'log_prob', 'ret'])

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
    # Set the model
    NN = REINFORCENet(feature_net, actor_net)
    return NN

  def reset_game(self):
    # Reset the game before a new episode
    self.state = self.state_normalizer(self.env.reset())
    self.next_state = None
    self.action = None
    self.log_prob = 0.0
    self.reward = None
    self.done = False
    self.episode_return = 0
    self.episode_step_count = 0    

  def save_experience(self, prediction):
    if self.reward is not None:
      prediction['mask'] = to_tensor(1-self.done, self.device)
      prediction['reward'] = to_tensor(self.reward, self.device)
    self.replay.add(prediction)

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
    self.save_episode_result('Train')
    self.save_episode_result('Test')

  def run_episode(self, mode, render):
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
      self.episode_step_count += 1
      if mode == 'Train':
        self.step_count += 1
        # Save experience
        self.save_experience(prediction)        
      # Update state
      self.state = self.next_state
    # End of one episode
    self.save_episode_result(mode)
    if mode == 'Train':
      self.episode_count += 1
      # Update policy
      self.learn()
      # Reset storage
      self.replay.empty()
    # Reset environment
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
    if mode == 'Test' or self.episode_count % self.display_interval == 0 or self.step_count >= self.train_steps:
      # Save result to files
      result = pd.DataFrame(self.result[mode])
      result['Env'] = result['Env'].astype('category')
      result['Agent'] = result['Agent'].astype('category')
      result.to_feather(self.log_path[mode])
      # Show log
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
    # Compute return
    self.replay.placeholder(self.episode_step_count)
    ret = torch.tensor(0.0)
    for i in reversed(range(self.episode_step_count)):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      self.replay.ret[i] = ret.detach()
    # Get training data
    entries = self.replay.get(['log_prob', 'ret'], self.episode_step_count)
    # Compute loss
    actor_loss = -(entries.log_prob * entries.ret).mean()
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: actor_loss={actor_loss.item()}')
    # Take an optimization step
    self.optimizer['actor'].zero_grad()
    actor_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
    self.optimizer['actor'].step()

  def get_action_size(self):
    if isinstance(self.env.action_space, Discrete):
      self.action_type = 'DISCRETE'
      return self.env.action_space.n
    elif isinstance(self.env.action_space, Box):
      self.action_type = 'CONTINUOUS'
      # Set the maximum abs value of action space, used for SAC.
      self.action_lim = max(max(abs(self.env.action_space.low)), max(self.env.action_space.high))  
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