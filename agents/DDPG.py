from agents.SAC import *


class DDPG(SAC):
  '''
  Implementation of DDPG (Deep Deterministic Policy Gradient)
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
    actor_net = MLPDeterministicActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[self.action_size], hidden_act=self.cfg['hidden_act'])
    # Set critic network
    critic_net = MLPQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = ActorQCriticNet(feature_net, actor_net, critic_net)
    return NN

  def get_action(self, mode='Train'):
    '''
    Pick an action from policy network
    '''
    if self.step_count <= self.cfg['exploration_steps']:
      prediction = {'action': torch.as_tensor(self.env[mode].action_space.sample())}
    else:
      state = to_tensor(self.state[mode], self.device)
      prediction = self.network(state)
    # Add noise
    if mode == 'Train': 
      prediction['action'] += self.cfg['action_noise'] * torch.randn(self.action_size)
    return prediction

  def compute_actor_loss(self, batch):
    q = self.network(batch.state)['q']
    actor_loss = - q.mean()
    return actor_loss

  def compute_critic_loss(self, batch):
    q = self.comput_q(batch) # Compute q
    q_target = self.compute_q_target(batch) # Compute q target
    critic_loss = (q - q_target).pow(2).mean()
    return critic_loss

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_next = self.network_target(batch.next_state)['q']
      q_target = batch.reward + self.discount * batch.mask * q_next
    return q_target

  def comput_q(self, batch):
    q = self.network.get_q(batch.state, batch.action)
    return q