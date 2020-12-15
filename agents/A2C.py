from agents.REINFORCEWithBaseline import *


class A2C(REINFORCEWithBaseline):
  '''
  Implementation of A2C (Syncrhonous Advantage Actor Critic, a synchronous deterministic version of A3C)
  '''
  def __init__(self, cfg):
    cfg.setdefault('test_per_episodes', 0) # To avoid an error of copying test_per_episodes in REINFORCEWithBaseline
    super().__init__(cfg)
    self.test_per_epochs = cfg['test_per_epochs']
    self.steps_per_epoch = cfg['steps_per_epoch']
    # Set replay buffer
    self.replay = FiniteReplay(self.steps_per_epoch+1, keys=['reward', 'mask', 'v', 'log_prob', 'ret', 'adv'])

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
      if mode == 'Train' and self.test_per_epochs > 0 and self.epoch_count % self.test_per_epochs == 0:
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
      for _ in range(self.steps_per_epoch):
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
          self.episode_count += 1
          self.save_episode_result(mode)
          self.reset_game(mode)
      if self.cfg['agent']['name'] in ['A2C', 'PPO']:
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
    # Save reward, mask, v, log_prob
    mode = 'Train'
    if self.reward[mode] is not None:
      prediction = {
        'reward': to_tensor(self.reward[mode], self.device),
        'mask': to_tensor(1-self.done[mode], self.device),
        'v': prediction['prediction'],
        'log_prob': prediction['log_prob']
      }
      self.replay.add(prediction)
    else:
      self.replay.add({'v':prediction['v']})

  def learn(self):
    mode = 'Train'
    # Compute return and advantage
    adv = torch.tensor(0.0)
    ret = self.replay.v[-1].detach()
    for i in reversed(range(self.steps_per_epoch)):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      if self.cfg['gae'] < 0:
        adv = ret - self.replay.v[i].detach()
      else:
        td_error = self.replay.reward[i] + self.discount * self.replay.mask[i] * self.replay.v[i+1] - self.replay.v[i]
        adv = self.discount * self.cfg['gae'] * self.replay.mask[i] * adv + td_error
      self.replay.adv[i] = adv.detach()
      self.replay.ret[i] = ret.detach()
    # Get training data
    entries = self.replay.get(['log_prob', 'v', 'ret', 'adv'], self.steps_per_epoch)
    # # Normalize advantages
    # entries.adv.copy_((entries.adv - entries.adv.mean()) / entries.adv.std())
    # Compute losses
    actor_loss = -(entries.log_prob * entries.adv).mean()
    critic_loss = (entries.ret - entries.v).pow(2).mean()
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
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