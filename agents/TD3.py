from agents.DDPG import *


class TD3(DDPG):
  '''
  Implementation of TD3 (Twin Delayed Deep Deterministic Policy Gradients)
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
    actor_net = MLPDeterministicActor(action_lim=self.action_lim, layer_dims=[input_size]+self.cfg['hidden_layers']+[self.action_size], hidden_activation=self.hidden_activation)
    # Set critic network
    critic_net = MLPDoubleQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
    # Set the model
    NN = TD3Net(feature_net, actor_net, critic_net)
    return NN

  def compute_actor_loss(self, batch):
    q1 = self.network(batch.state)['q1']
    actor_loss = - q1.mean()
    return actor_loss

  def compute_critic_loss(self, batch):
    q1, q2 = self.comput_q(batch) # Compute q
    q_target = self.compute_q_target(batch) # Compute q target
    critic_loss = ((q1-q_target).pow(2) + (q2-q_target).pow(2)).mean()
    return critic_loss

  def compute_q_target(self, batch):
    with torch.no_grad():
      action = self.network_target(batch.next_state)['action']
      noise = torch.randn_like(action).mul(self.cfg['target_noise'])
      noise = noise.clamp(-self.cfg['noise_clip'], self.cfg['noise_clip'])
      action = (action + noise).clamp(self.action_min, self.action_max)
      q1, q2 = self.network_target.get_q(batch.next_state, action)
      q_next = torch.min(q1, q2)
      q_target = batch.reward + self.discount * batch.mask * q_next
    return q_target

  def comput_q(self, batch):
    q1, q2 = self.network.get_q(batch.state, batch.action)
    return q1, q2