from agents.PPO import *


class OnRPG4(PPO):
  '''
  PPO style 2: update actor only once for each epoch
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
    self.replay = FiniteReplay(self.steps_per_epoch, keys=['state', 'action', 'reward', 'next_state', 'mask', 'log_prob', 'step'])

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
    actor_net = MLPSquashedGaussianActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_act=self.cfg['hidden_act'])
    # Set critic network (state value)
    critic_net = MLPCritic(layer_dims=[input_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set reward network
    reward_net = MLPQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorVCriticRewardNet(feature_net, actor_net, critic_net, reward_net)
    return NN

  def save_experience(self, prediction):
    # Save state, action, reward, next_state, mask, log_prob
    mode = 'Train'
    prediction = {
      'state': to_tensor(self.state[mode], self.device),
      'action': to_tensor(self.action[mode], self.device),
      'reward': to_tensor(self.reward[mode], self.device),
      'next_state': to_tensor(self.next_state[mode], self.device),
      'mask': to_tensor(1-self.done[mode], self.device),
      'step': to_tensor(self.episode_step_count[mode]-1, self.device),
      'log_prob': prediction['log_prob']
    }
    self.replay.add(prediction)

  def learn(self):
    mode = 'Train'
    # Get training data and **detach** (IMPORTANT: we don't optimize old parameters)
    entries = self.replay.get(['state', 'action', 'reward', 'mask', 'next_state', 'log_prob', 'step'], self.steps_per_epoch, detach=True)
    # Optimize for multiple epochs
    for _ in range(self.cfg['optimize_epochs']):
      batch_idxs = generate_batch_idxs(len(entries.log_prob), self.cfg['batch_size'])
      for batch_idx in batch_idxs:
        batch_idx = to_tensor(batch_idx, self.device).long()
        # Take an optimization step for critic
        v = self.network.get_state_value(entries.state[batch_idx])
        v_next = self.network.get_state_value(entries.next_state[batch_idx]).detach()
        critic_loss = (entries.reward[batch_idx] + self.discount*entries.mask[batch_idx]*v_next - v).pow(2).mean()
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.gradient_clip > 0:
          nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
        self.optimizer['critic'].step()
        # Take an optimization step for reward
        reward = self.network.get_reward(entries.state[batch_idx], entries.action[batch_idx])
        reward_loss = (reward - entries.reward[batch_idx]).pow(2).mean()
        self.optimizer['reward'].zero_grad()
        reward_loss.backward()
        if self.gradient_clip > 0:
          nn.utils.clip_grad_norm_(self.network.reward_params, self.gradient_clip)
        self.optimizer['reward'].step()
      # Take an optimization step for actor
      prediction = self.network(entries.state, entries.action)
      # Freeze the reward network to avoid computing gradients for it
      for p in self.network.reward_net.parameters():
        p.requires_grad = False
      repara_action = self.network.get_repara_action(entries.state, entries.action)
      reward = self.network.get_reward(entries.state, repara_action)
      v_next = self.network.get_state_value(entries.next_state).detach()
      # discounts = to_tensor([self.discount**i for i in entries.step], self.device)
      # actor_loss = -(discounts * (reward + self.discount*entries.mask*v_next*prediction['log_prob'])).mean()
      actor_loss = -(reward + self.discount*entries.mask*v_next*prediction['log_prob']).mean()
      self.optimizer['actor'].zero_grad()
      actor_loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      self.optimizer['actor'].step()
      # Unfreeze the reward network
      for p in self.network.reward_net.parameters():
        p.requires_grad = True
    # Log
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
      self.logger.add_scalar(f'reward_loss', reward_loss.item(), self.step_count)