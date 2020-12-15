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
    self.env = {
      'Train': make_env(cfg['env']['name'], max_episode_steps=int(cfg['env']['max_episode_steps'])),
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
    # Create policy network
    self.network = self.createNN(cfg['env']['input_type']).to(self.device)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs'])
    }
    # Set replay buffer
    self.replay = InfiniteReplay(keys=['reward', 'mask', 'log_prob', 'ret'])
    # Set log dict
    for key in ['state', 'next_state', 'action', 'log_prob', 'reward', 'done', 'episode_return', 'episode_step_count']:
      setattr(self, key, {'Train': None, 'Test': None})

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers']
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env['Train'].game.state_shape()[2], feature_dim=layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    elif input_type == 'feature':
      layer_dims = [self.state_size] + self.cfg['hidden_layers']
      feature_net = nn.Identity()
    # Set actor network
    if self.action_type == 'DISCRETE':
      actor_net = MLPCategoricalActor(layer_dims=layer_dims+[self.action_size], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    elif self.action_type == 'CONTINUOUS':
      actor_net = MLPGaussianActor(action_lim=self.action_lim, layer_dims=layer_dims+[self.action_size], hidden_act=self.cfg['hidden_act'])
    # Set the model
    NN = REINFORCENet(feature_net, actor_net)
    return NN

  def reset_game(self, mode):
    # Reset the game before a new episode
    self.state[mode] = self.state_normalizer(self.env[mode].reset())
    self.next_state[mode] = None
    self.action[mode] = None
    self.log_prob[mode] = 0.0
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
      if mode == 'Train' and self.test_per_episodes > 0 and self.episode_count % self.test_per_episodes == 0:
        mode = 'Test'
      else:
        mode = 'Train'
      # Set network back to training/evaluation mode
      self.set_net_mode(mode)
      # Run for one episode
      self.run_episode(mode, render)

  def run_episode(self, mode, render):
    while not self.done[mode]:
      prediction = self.get_action(mode)
      self.action[mode] = to_numpy(prediction['action'])
      # Clip the action
      if self.action_type == 'CONTINUOUS':
        action = np.clip(self.action[mode], self.action_min, self.action_max)
      else:
        action = self.action[mode]
      if render:
        self.env[mode].render()
      # Take a step
      self.next_state[mode], self.reward[mode], self.done[mode], _ = self.env[mode].step(action)
      self.next_state[mode] = self.state_normalizer(self.next_state[mode])
      self.reward[mode] = self.reward_normalizer(self.reward[mode])
      self.episode_return[mode] += self.reward[mode]
      self.episode_step_count[mode] += 1
      if mode == 'Train':
        self.step_count += 1
        # Save experience
        self.save_experience(prediction)        
      # Update state
      self.state[mode] = self.next_state[mode]
    # End of one episode
    self.save_episode_result(mode)
    if mode == 'Train':
      self.episode_count += 1
      # Update policy
      self.learn()
      # Reset storage
      self.replay.clear()
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
    Pick an action from policy network
    '''
    state = to_tensor(self.state[mode], self.device)
    deterministic = True if mode == 'Test' else False
    prediction = self.network(state, deterministic=deterministic)
    return prediction

  def save_experience(self, prediction):
    # Save reward, mask, log_prob
    mode = 'Train'
    prediction = {
      'reward': to_tensor(self.reward[mode], self.device),
      'mask': to_tensor(1-self.done[mode], self.device),
      'log_prob': prediction['log_prob']
    }
    self.replay.add(prediction)

  def learn(self):
    mode = 'Train'
    # Compute return
    self.replay.placeholder(self.episode_step_count[mode])
    ret = torch.tensor(0.0)
    for i in reversed(range(self.episode_step_count[mode])):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      self.replay.ret[i] = ret.detach()
    # Get training data
    entries = self.replay.get(['log_prob', 'ret'], self.episode_step_count[mode])
    # Compute loss
    actor_loss = -(entries.log_prob * entries.ret).mean()
    # Take an optimization step for actor
    self.optimizer['actor'].zero_grad()
    actor_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
    self.optimizer['actor'].step()
    # Log
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: actor_loss={actor_loss.item()}')

  def get_action_size(self):
    mode = 'Train'
    if isinstance(self.env[mode].action_space, Discrete):
      self.action_type = 'DISCRETE'
      return self.env[mode].action_space.n
    elif isinstance(self.env[mode].action_space, Box):
      self.action_type = 'CONTINUOUS'
      # Set the minimum/maximum/limit values of the action spaces
      self.action_min = min(self.env[mode].action_space.low)
      self.action_max = max(self.env[mode].action_space.high)
      self.action_lim = max(abs(self.action_min), self.action_max)
      return self.env[mode].action_space.shape[0]
    else:
      raise ValueError('Unknown action type.')
    
  def get_state_size(self):
    mode = 'Train'
    return int(np.prod(self.env[mode].observation_space.shape))

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