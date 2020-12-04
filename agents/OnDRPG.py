from agents.OnRPG import *


class OnDRPG(OnRPG):
  '''
  Implementation of OnDRPG (On-policy Double Reward Policy Gradient)
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
    critic_net = MLPDoubleQCritic(layer_dims=[input_size+self.action_size]+self.cfg['hidden_layers']+[1], hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'])
    # Set the model
    NN = DRPGNet(feature_net, actor_net, critic_net)
    return NN
  
  def save_experience(self, prediction):
    if 'reward1' in prediction.keys():
      del prediction['reward1'], prediction['reward2']
    super().save_experience(prediction)
  
  def compute_critic_loss(self, batch):
    true_reward = batch.reward
    r1, r2 = self.network.get_reward(batch.state, batch.action)
    predicted_reward = torch.min(r1, r2)
    critic_loss = (predicted_reward - true_reward).pow(2).mean()
    return critic_loss

  def compute_actor_loss(self, batch):
    r1, r2 = self.network.get_reward(batch.state, batch.action)
    predicted_reward = torch.min(r1, r2)
    discounts = to_tensor([self.discount**i for i in range(predicted_reward.size(0))], self.device)
    actor_loss = - (predicted_reward * discounts).sum()
    return actor_loss