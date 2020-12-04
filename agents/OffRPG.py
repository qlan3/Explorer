from agents.SAC import *


class OffRPG(SAC):
  '''
  Implementation of OffRPG (Off-policy Reward Policy Gradient)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs']),
      'critic':  getattr(torch.optim, cfg['optimizer']['name'])(self.network.critic_params, **cfg['optimizer']['critic_kwargs'])
    }
    # Set replay buffer
    self.replay = FiniteReplay(cfg['memory_size'], keys=['state', 'action', 'reward'])
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
    assert self.action_type == 'CONTINUOUS', f"{self.cfg['agent']['name']} only supports continous action spaces."
    actor_net = MLPRepGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_act=self.cfg['hidden_act'])
    # Set critic network
    critic_net = MLPQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = RPGNet(feature_net, actor_net, critic_net)
    return NN

  def save_experience(self, prediction):
    mode = 'Train'
    prediction = {
      'state': to_tensor(self.state[mode], self.device),
      'action': to_tensor(self.action[mode], self.device),
      'reward': to_tensor(self.reward[mode], self.device)
    }
    self.replay.add(prediction)

  def learn(self):
    mode = 'Train'
    batch = self.replay.sample(['state', 'action', 'reward'], self.cfg['batch_size'])
    # Compute critic loss
    critic_loss = self.compute_critic_loss(batch)
    # Take an optimization step for critic
    self.optimizer['critic'].zero_grad()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['critic'].step()
    
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['actor_update_frequency'] == 0:
      # Freeze reward network to avoid computing gradients for them
      for p in self.network.critic_net.parameters():
        p.requires_grad = False
      # Compute actor loss
      actor_loss = self.compute_actor_loss(batch)
      # Take an optimization step for actor
      self.optimizer['actor'].zero_grad()
      actor_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      self.optimizer['actor'].step()
      # Unfreeze reward network
      for p in self.network.critic_net.parameters():
        p.requires_grad = True
      # Log
      if self.show_tb:
        self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
        self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
      self.logger.debug(f'Step {self.step_count}: actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}')

  def compute_critic_loss(self, batch):
    true_reward = batch.reward
    predicted_reward = self.network.get_reward(batch.state, batch.action)
    critic_loss = (predicted_reward - true_reward).pow(2).mean()
    return critic_loss

  def compute_actor_loss(self, batch):
    prediction = self.network(batch.state)
    predicted_reward = prediction['reward']
    actor_loss = - predicted_reward.mean()
    return actor_loss