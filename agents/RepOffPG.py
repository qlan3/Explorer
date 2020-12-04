from agents.DDPG import *


class RepOffPG(DDPG):
  '''
  Implementation of RepOffPG (Reparameterization Off-Policy Gradient), almost the same as SVG(0).
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

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
    NN = RepActorCriticNet(feature_net, actor_net, critic_net)
    return NN

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    '''
    if self.step_count <= self.cfg['exploration_steps']:
      prediction = {'action': torch.as_tensor(self.env[mode].action_space.sample())}
    else:
      deterministic = True if mode=='Test' else False
      state = to_tensor(self.state[mode], self.device)
      prediction = self.network(state, deterministic=deterministic)
    # Clip the action
    prediction['action'] = torch.clamp(prediction['action'], self.action_min, self.action_max)
    return prediction

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_next = self.network_target(batch.next_state, deterministic=True)['q']
      q_target = batch.reward + self.discount * batch.mask * q_next
    return q_target