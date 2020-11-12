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
        feature_net = Conv2d_MinAtar(in_channels=self.env.game.state_shape()[2], feature_dim=input_size)
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=input_size)
    elif input_type == 'feature':
      input_size = self.state_size
      feature_net = nn.Identity()
    # Set actor network
    assert self.action_type == 'CONTINUOUS', "SAC only supports continous action space."
    actor_net = MLPSquashedGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_activation=self.hidden_activation)
    # Set critic network
    critic_net = DoubleQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    # Set the model
    NN = SACNet(feature_net, actor_net, critic_net)
    return NN
  
  def save_experience(self, prediction): 
    experience = {
      'state': to_tensor(self.state, self.device),
      'action': prediction['action'].detach(),
      'next_state': to_tensor(self.next_state, self.device),
      'reward': to_tensor(self.reward, self.device),
      'mask': to_tensor(1-self.done, self.device)
    }
    self.replay.add(experience)

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
      if mode == 'Train':
        # Save experience
        self.save_experience(prediction)
        # Update policy
        if self.time_to_learn():
          self.learn()
        self.step_count += 1
      # Update state
      self.state = self.next_state
    # End of one episode
    self.save_episode_result(mode)
    # Reset environment
    self.reset_game()
    if mode == 'Train':
      self.episode_count += 1

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    '''
    if self.step_count <= self.cfg['exploration_steps']:
      prediction = {'action': torch.as_tensor(self.env.action_space.sample())}
    else:
      deterministic = True if mode=='Test' else False
      state = to_tensor(self.state, self.device)
      prediction = self.network(state, deterministic=deterministic)
    return prediction

  def time_to_learn(self):
    """
    Return boolean to indicate whether it is time to learn:
    - The agent is not on exploration stage
    - There are enough experiences in replay buffer
    """
    if self.step_count > self.cfg['exploration_steps'] and self.step_count > self.cfg['batch_size'] and self.step_count % self.cfg['network_update_frequency'] == 0:
      return True
    else:
      return False

  def learn(self):
    for i in range(self.cfg['network_update_frequency']):
      batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
      # Compute critic loss
      q1, q2 = self.comput_q(batch) # Compute q
      q_target = self.compute_q_target(batch) # Compute q target
      critic_loss = ((q1-q_target)**2 + (q2-q_target)**2).mean()
      # Take an optimization step for critic
      self.optimizer['critic'].zero_grad()
      critic_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
      self.optimizer['critic'].step()
      
      # Freeze Q-networks to avoid computing gradients for them
      for p in self.network.critic_net.parameters():
        p.requires_grad = False
      # Compute actor loss
      prediction = self.network(batch.state)
      q1, q2, log_prob = prediction['q1'], prediction['q2'], prediction['log_prob']
      q_min = torch.min(q1, q2)
      actor_loss = (self.cfg['alpha'] * log_prob - q_min).mean()
      # Take an optimization step for actor
      self.optimizer['actor'].zero_grad()
      actor_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      self.optimizer['actor'].step()
      # Unfreeze Q-networks
      for p in self.network.critic_net.parameters():
        p.requires_grad = True
      
      # Update target networks by polyak averaging (soft update)
      if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
        with torch.no_grad():
          for p, p_target in zip(self.network.parameters(), self.network_target.parameters()):
            p_target.data.mul_(self.cfg['polyak'])
            p_target.data.add_((1-self.cfg['polyak'])*p.data)

  def compute_q_target(self, batch):
    with torch.no_grad():
      # Sample action from *current* policy
      prediction = self.network(batch.next_state)
      action, log_prob = prediction['action'], prediction['log_prob']
      prediction_target = self.network_target(batch.next_state, action)
      q_next = torch.min(prediction_target['q1'], prediction_target['q2'])
      q_target = batch.reward + self.discount * batch.mask * (q_next - self.cfg['alpha'] * log_prob)
    return q_target

  def comput_q(self, batch):
    prediction = self.network(batch.state, batch.action)
    return prediction['q1'], prediction['q2']