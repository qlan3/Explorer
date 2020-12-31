from agents.REINFORCE import *


class ActorCritic(REINFORCE):
  '''
  Implementation of Actor-Critic (state value function)
    - get REINFORCE with baseline when cfg['gae'] < 0
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs']),
      'critic':  getattr(torch.optim, cfg['optimizer']['name'])(self.network.critic_params, **cfg['optimizer']['critic_kwargs'])
    }
    # Set replay buffer
    self.replay = FiniteReplay(self.cfg['steps_per_epoch']+1, keys=['reward', 'mask', 'v', 'log_pi', 'ret', 'adv'])

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers']
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env[mode].game.state_shape()[2], feature_dim=layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    elif input_type == 'feature':
      layer_dims = [self.state_size] + self.cfg['hidden_layers']
      feature_net = nn.Identity()
    # Set actor network
    if self.action_type == 'DISCRETE':
      actor_net = MLPCategoricalActor(layer_dims=layer_dims+[self.action_size], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    elif self.action_type == 'CONTINUOUS':
      actor_net = MLPGaussianActor(action_lim=self.action_lim ,layer_dims=layer_dims+[self.action_size], hidden_act=self.cfg['hidden_act'])
    # Set critic network
    critic_net = MLPCritic(layer_dims=layer_dims+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorVCriticNet(feature_net, actor_net, critic_net)
    return NN
    
  def run_steps(self, render=False):
    # Run for multiple episodes
    self.step_count = 0
    self.episode_count = 0
    self.epoch_count = 0
    self.result = {'Train': [], 'Test': []}
    self.episode_return_list = {'Train': [], 'Test': []}
    mode = 'Train'
    self.start_time = time.time()
    self.reset_game('Train')
    self.reset_game('Test')
    while self.step_count < self.train_steps:
      if mode == 'Train' and self.cfg['test_per_epochs'] > 0 and self.epoch_count % self.cfg['test_per_epochs'] == 0:
        mode = 'Test'
      else:
        mode = 'Train'
        self.epoch_count += 1
      # Set network back to training/evaluation mode
      self.set_net_mode(mode)
      # Run for one epoch
      self.run_epoch(mode, render)

  def run_epoch(self, mode, render):
    if mode == 'Train':
      # Run for one epoch
      for _ in range(self.cfg['steps_per_epoch']):
        prediction = self.get_action(mode)
        self.action[mode] = to_numpy(prediction['action'])
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
        self.step_count += 1
        # Save experience
        self.save_experience(prediction)
        # Update state
        self.state[mode] = self.next_state[mode]
        # End of one episode
        if self.done[mode]:
          self.save_episode_result(mode)
          self.episode_count += 1
          self.reset_game(mode)
      if self.cfg['agent']['name'] in ['ActorCritic', 'PPO']:
        prediction = self.get_action(mode)
        self.save_experience(prediction)
      # Update policy
      self.learn()
      # Reset storage
      self.replay.clear()
    elif mode == 'Test':
      # Run for one episode
      while not self.done[mode]:
        prediction = self.get_action(mode)
        self.action[mode] = to_numpy(prediction['action'])
        if render:
          self.env[mode].render()
        # Take a step
        self.next_state[mode], self.reward[mode], self.done[mode], _ = self.env[mode].step(self.action[mode])
        self.next_state[mode] = self.state_normalizer(self.next_state[mode])
        self.reward[mode] = self.reward_normalizer(self.reward[mode])
        self.episode_return[mode] += self.reward[mode]
        # Update state
        self.state[mode] = self.next_state[mode]
      # End of one episode
      self.save_episode_result(mode)
      self.reset_game(mode)

  def save_experience(self, prediction):
    # Save reward, mask, v, log_pi
    mode = 'Train'
    if self.reward[mode] is not None:
      prediction = {
        'reward': to_tensor(self.reward[mode], self.device),
        'mask': to_tensor(1-self.done[mode], self.device),
        'v': prediction['v'],
        'log_pi': prediction['log_pi']
      }
      self.replay.add(prediction)
    else:
      self.replay.add({'v':prediction['v']})

  def learn(self):
    mode = 'Train'
    # Compute return and advantage
    adv = torch.tensor(0.0)
    ret = self.replay.v[-1].detach()
    for i in reversed(range(self.cfg['steps_per_epoch'])):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      if self.cfg['gae'] < 0:
        adv = ret - self.replay.v[i].detach()
      else:
        td_error = self.replay.reward[i] + self.discount * self.replay.mask[i] * self.replay.v[i+1] - self.replay.v[i]
        adv = self.discount * self.cfg['gae'] * self.replay.mask[i] * adv + td_error
      print('adv len: ', len(self.replay.adv))
      self.replay.adv[i] = adv.detach()
      self.replay.ret[i] = ret.detach()
    # Get training data
    entries = self.replay.get(['log_pi', 'v', 'ret', 'adv'], self.cfg['steps_per_epoch'])
    # # Normalize advantages
    # entries.adv.copy_((entries.adv - entries.adv.mean()) / entries.adv.std())
    # Compute losses
    actor_loss = -(entries.log_pi * entries.adv).mean()
    critic_loss = (entries.ret - entries.v).pow(2).mean()
    # Take an optimization step for actor
    self.optimizer['actor'].zero_grad()
    actor_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
    self.optimizer['actor'].step()
    # Take an optimization step for critic
    self.optimizer['critic'].zero_grad()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['critic'].step()
    # Log
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
      self.logger.add_scalar(f'v', entries.v.mean().item(), self.step_count)
      self.logger.add_scalar(f'log_pi', entries.log_pi.mean().item(), self.step_count)