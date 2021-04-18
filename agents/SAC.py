from agents.REINFORCE import *


class SAC(REINFORCE):
  '''
  Implementation of SAC (Soft Actor-Critic)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Create target policy network
    self.network_target = self.createNN(cfg['env']['input_type']).to(self.device)
    self.network_target.load_state_dict(self.network.state_dict())
    # Freeze target policy network (only updated via polyak averaging)
    for p in self.network_target.parameters():
      p.requires_grad = False
    self.network_target.eval()
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs']),
      'critic':  getattr(torch.optim, cfg['optimizer']['name'])(self.network.critic_params, **cfg['optimizer']['critic_kwargs'])
    }
    # Set replay buffer
    self.replay = FiniteReplay(cfg['memory_size'], keys=['state', 'action', 'next_state', 'reward', 'mask'])
    self.cfg['exploration_steps'] = int(self.cfg['exploration_steps'])

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      input_size = self.cfg['feature_dim']
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env[mode].game.state_shape()[2], feature_dim=input_size)
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=input_size)
    elif input_type == 'feature':
      input_size = self.state_size
      feature_net = nn.Identity()
    # Set actor network
    assert self.action_type == 'CONTINUOUS', f'{self.agent_name} only supports continous action spaces.'
    actor_net = MLPSquashedGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_act=self.cfg['hidden_act'], rsample=True)
    # Set critic network
    critic_net = MLPDoubleQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorDoubleQCriticNet(feature_net, actor_net, critic_net)
    return NN
  
  def save_experience(self, prediction):
    # Save state, action, next_state, reward, mask
    mode = 'Train'
    prediction = {
      'state': to_tensor(self.state[mode], self.device),
      'action': to_tensor(self.action[mode], self.device),
      'next_state': to_tensor(self.next_state[mode], self.device),
      'reward': to_tensor(self.reward[mode], self.device),
      'mask': to_tensor(1-self.done[mode], self.device)
    }
    self.replay.add(prediction)

  def run_episode(self, mode, render):
    while not self.done[mode]:
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
      if mode == 'Train':
        # Save experience
        self.save_experience(prediction)
        # Update policy
        if self.time_to_learn():
          self.learn()
        self.step_count += 1
      # Update state
      self.state[mode] = self.next_state[mode]
    # End of one episode
    self.save_episode_result(mode)
    # Reset environment
    self.reset_game(mode)
    if mode == 'Train':
      self.episode_count += 1

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    '''
    state = to_tensor(self.state[mode], self.device)
    if self.step_count <= self.cfg['exploration_steps']:
      action = to_tensor(self.env[mode].action_space.sample(), self.device)
      prediction = self.network(state, action=action)
    else:
      deterministic = True if mode == 'Test' else False
      prediction = self.network(state, deterministic=deterministic)
    return prediction

  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
    - The agent is not on exploration stage
    - It is time to update network
    """
    if self.step_count > self.cfg['exploration_steps'] and self.step_count % self.cfg['network_update_frequency'] == 0:
      return True
    else:
      return False

  def learn(self):
    mode = 'Train'
    batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    # Take an optimization step for critic
    critic_loss = self.compute_critic_loss(batch)
    self.optimizer['critic'].zero_grad()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['critic'].step()
    # Take an optimization step for actor
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['actor_update_frequency'] == 0:
      # Freeze Q-networks to avoid computing gradients for them
      for p in self.network.critic_net.parameters():
        p.requires_grad = False
      # Compute actor loss
      actor_loss = self.compute_actor_loss(batch)
      self.optimizer['actor'].zero_grad()
      actor_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      self.optimizer['actor'].step()
      # Unfreeze Q-networks
      for p in self.network.critic_net.parameters():
        p.requires_grad = True
      # Update target networks by polyak averaging (soft update)
      self.soft_update(self.network, self.network_target)
      # Log
      if self.show_tb:
        self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
        self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)

  def compute_actor_loss(self, batch):
    prediction = self.network(batch.state)
    q1, q2, log_pi = prediction['q1'], prediction['q2'], prediction['log_pi']
    q_min = torch.min(q1, q2)
    actor_loss = (self.cfg['alpha'] * log_pi - q_min).mean()
    return actor_loss

  def compute_critic_loss(self, batch):
    q1, q2 = self.compute_q(batch) # Compute q
    q_target = self.compute_q_target(batch) # Compute q target
    critic_loss = ((q1-q_target).pow(2) + (q2-q_target).pow(2)).mean()
    return critic_loss
    
  def compute_q_target(self, batch):
    with torch.no_grad():
      # Sample action from *current* policy
      prediction = self.network(batch.next_state)
      action, log_pi = prediction['action'], prediction['log_pi']
      q1, q2 = self.network_target.get_q(batch.next_state, action)
      q_next = torch.min(q1, q2)
      q_target = batch.reward + self.discount * batch.mask * (q_next - self.cfg['alpha'] * log_pi)
    return q_target

  def compute_q(self, batch):
    q1, q2 = self.network.get_q(batch.state, batch.action)
    return q1, q2

  def soft_update(self, network, network_target):
    with torch.no_grad():
      for p, p_target in zip(self.network.parameters(), self.network_target.parameters()):
        p_target.data.mul_(self.cfg['polyak'])
        p_target.data.add_((1-self.cfg['polyak'])*p.data)