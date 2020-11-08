from agents.REINFORCEWithBaseline import *


class A2C(REINFORCEWithBaseline):
  '''
  Implementation of A2C (Syncrhonous Advantage Actor Critic, a synchronous deterministic version of A3C)
  '''
  def __init__(self, cfg):
    cfg['test_per_episodes'] = -1 # To avoid an error of copying test_per_episodes in REINFORCEWithBaseline
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
    self.reset_game()
    while self.step_count < self.train_steps:
      if mode == 'Train' and self.epoch_count % self.test_per_epochs == 0:
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
      # Update policy
      self.learn()
      # Reset storage
      self.replay.empty()
    elif mode == 'Test':
      # Run for one episode
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
      self.save_episode_result(mode)
      self.reset_game()

  def learn(self):
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