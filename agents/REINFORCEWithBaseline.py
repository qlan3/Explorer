from agents.REINFORCE import *


class REINFORCEWithBaseline(REINFORCE):
  '''
  Implementation of REINFORCE with baseline (state value function)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set optimizer
    self.optimizer = {
      'actor': getattr(torch.optim, cfg['optimizer']['name'])(self.network.actor_params, **cfg['optimizer']['actor_kwargs']),
      'critic':  getattr(torch.optim, cfg['optimizer']['name'])(self.network.critic_params, **cfg['optimizer']['critic_kwargs'])
    }
    # Set replay buffer
    self.replay = InfiniteReplay(keys=['reward', 'mask', 'v', 'log_prob', 'adv'])
  
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
    NN = ActorCriticNet(feature_net, actor_net, critic_net)
    return NN

  def learn(self):
    mode = 'Train'
    # Compute advantage
    self.replay.placeholder(self.episode_step_count[mode])
    ret = torch.tensor(0.0)
    for i in reversed(range(self.episode_step_count[mode])):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      self.replay.adv[i] = ret.detach() - self.replay.v[i]
    # Get training data
    entries = self.replay.get(['log_prob', 'adv'], self.episode_step_count[mode])
    # # Normalize advantages
    # entries.adv.copy_((entries.adv - entries.adv.mean()) / entries.adv.std())
    # Compute loss
    actor_loss = -(entries.log_prob * entries.adv.detach()).mean()
    critic_loss = entries.adv.pow(2).mean()
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