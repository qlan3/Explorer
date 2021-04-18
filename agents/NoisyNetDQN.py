from agents.DQN import *


class NoisyNetDQN(DQN):
  '''
  Implementation of NoisyNet DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.Q_net_target[0].train() # Set target network to training mode

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers'] + [self.action_size]
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env['Train'].game.state_shape()[2], feature_dim=layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    elif input_type == 'feature':
      layer_dims = [self.state_size] + self.cfg['hidden_layers'] + [self.action_size]
      feature_net = nn.Identity()
    # Set value network
    assert self.action_type == 'DISCRETE', f'{self.agent_name} only supports discrete action spaces.'
    value_net = NoisyMLPCritic(layer_dims=layer_dims, hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    NN = DQNNet(feature_net, value_net)
    return NN

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    if mode == 'Train':
      self.Q_net[0].value_net.reset_noise()
    q_values = self.get_action_selection_q_values(state)
    action = np.argmax(q_values)
    return action

  def compute_q_target(self, batch):
    self.Q_net_target[0].value_net.reset_noise()
    return super().compute_q_target(batch)
  
  def compute_q(self, batch):
    self.Q_net[0].value_net.reset_noise()
    return super().compute_q(batch)