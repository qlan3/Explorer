from agents.A2C import *


class RepOnPG(A2C):
  '''
  Implementation of RepOnPG (Reparameterization On-Policy Gradient)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set replay buffer
    self.replay = FiniteReplay(self.steps_per_epoch+1, keys=['state', 'action', 'reward', 'mask', 'q', 'q_target'])

  def save_experience(self, prediction):
    # Save state, action, reward, mask, q
    mode = 'Train'
    if self.reward[mode] is not None:
      prediction = {
        'state': to_tensor(self.state[mode], self.device),
        'action': to_tensor(self.action[mode], self.device),
        'reward': to_tensor(self.reward[mode], self.device),
        'mask': to_tensor(1-self.done[mode], self.device),
        'q': prediction['prediction']
      }
      self.replay.add(prediction)
    else:
      self.replay.add({'q': prediction['q']})

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
    actor_net = MLPStdGaussianResActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[2*self.action_size], hidden_act=self.cfg['hidden_act'])
    # Set critic network
    critic_net = MLPQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorQCriticNet(feature_net, actor_net, critic_net)
    return NN

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    '''
    state = to_tensor(self.state[mode], self.device)
    deterministic = True if mode == 'Test' else False
    prediction = self.network(state, deterministic=deterministic)
    return prediction

  def learn(self):
    mode = 'Train'
    # Compute q target
    for i in range(self.steps_per_epoch):
      q_target = self.replay.reward[i] + self.discount * self.replay.mask[i] * self.replay.q[i+1]
      self.replay.q_target[i] = q_target.detach()
    # Get training data
    batch = self.replay.get(['state', 'action', 'q', 'q_target'], self.steps_per_epoch)
    # Take an optimization step for critic
    critic_loss = self.compute_critic_loss(batch)
    self.optimizer['critic'].zero_grad()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['critic'].step()
    # Freeze Q-networks to avoid computing gradients for them
    for p in self.network.critic_net.parameters():
      p.requires_grad = False
    # Take an optimization step for actor
    actor_loss = self.compute_actor_loss(batch)
    self.optimizer['actor'].zero_grad()
    actor_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
    self.optimizer['actor'].step()
    # Unfreeze Q-networks
    for p in self.network.critic_net.parameters():
      p.requires_grad = True
    # Log
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)

  def compute_actor_loss(self, batch):
    actor_loss = - batch.q.mean()
    return actor_loss

  def compute_critic_loss(self, batch):
    q = self.network.get_q(batch.state, batch.action)
    critic_loss = (q - batch.q_target).pow(2).mean()
    return critic_loss