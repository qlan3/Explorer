from agents.REINFORCE import *


class OnRPG32(REINFORCE):
  '''
  REINFORCE style
  Implementation of OnRPG (On-policy Reward Policy Gradient)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs']),
      'critic': getattr(torch.optim, cfg['optimizer']['name'])(self.network.critic_params, **cfg['optimizer']['critic_kwargs']),
      'reward': getattr(torch.optim, cfg['optimizer']['name'])(self.network.reward_params, **cfg['optimizer']['critic_kwargs'])
    }
    # Set replay buffer
    self.replay = InfiniteReplay(keys=['state', 'action', 'reward', 'next_state', 'mask', 'log_prob', 'step'])

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
    # actor_net = MLPStdGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_act=self.cfg['hidden_act'])
    actor_net = MLPSquashedGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_act=self.cfg['hidden_act'], rsample=True)
    # Set critic network (state value)
    critic_net = MLPCritic(layer_dims=[input_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set reward network
    reward_net = MLPQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorVCriticRewardNet(feature_net, actor_net, critic_net, reward_net)
    return NN

  def save_experience(self, prediction):
    # Save state, action, reward, next_state, mask, log_prob, step
    mode = 'Train'
    prediction = {
      'state': to_tensor(self.state[mode], self.device),
      'action': prediction['action'],
      'reward': to_tensor(self.reward[mode], self.device),
      'next_state': to_tensor(self.next_state[mode], self.device),
      'mask': to_tensor(1-self.done[mode], self.device),
      'step': to_tensor(self.episode_step_count[mode]-1, self.device)
    }
    prediction['log_prob'] = self.network.get_log_prob(prediction['state'], prediction['action'].detach())
    self.replay.add(prediction)

  def learn(self):
    mode = 'Train'
    # Get training data
    entries = self.replay.get(['state', 'action', 'reward', 'mask', 'next_state', 'log_prob', 'step'], self.episode_step_count[mode])
    v = self.network.get_state_value(entries.state)
    v_next = self.network.get_state_value(entries.next_state).detach()
    if self.episode_count % self.cfg['actor_update_frequency'] == 0:
      # Take an optimization step for actor
      predicted_reward = self.network.get_reward(entries.state, entries.action)
      # Freeze reward network to avoid computing gradients for it
      for p in self.network.reward_net.parameters():
        p.requires_grad = False
      # discounts = to_tensor([self.discount**i for i in entries.step], self.device)
      # actor_loss = -(discounts * (predicted_reward + self.discount*entries.mask*v_next*entries.log_prob)).mean()
      actor_loss = -(predicted_reward + self.discount*entries.mask*v_next*entries.log_prob).mean()
      self.optimizer['actor'].zero_grad()
      actor_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      self.optimizer['actor'].step()
      # Unfreeze reward network
      for p in self.network.reward_net.parameters():
        p.requires_grad = True
    # Take an optimization step for critic
    critic_loss = (entries.reward + self.discount*entries.mask*v_next - v).pow(2).mean()
    self.optimizer['critic'].zero_grad()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['critic'].step()
    # Take an optimization step for reward
    reward = self.network.get_reward(entries.state, entries.action.detach())
    reward_loss = (reward - entries.reward).pow(2).mean()
    self.optimizer['reward'].zero_grad()
    reward_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.reward_params, self.gradient_clip)
    self.optimizer['reward'].step()
    # Log
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
      self.logger.add_scalar(f'reward_loss', reward_loss.item(), self.step_count)
      self.logger.add_scalar(f'v_mean', v.mean().item(), self.step_count)