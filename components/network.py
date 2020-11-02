import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


activations = {
  'None': nn.Identity(),
  'ReLU': nn.ReLU(),
  'LeakyReLU': nn.LeakyReLU(),
  'Tanh': nn.Tanh(),
  'Sigmoid': nn.Sigmoid(),
  'Softmax-1': nn.Softmax(dim=-1),
  'Softmax0': nn.Softmax(dim=0),
  'Softmax1': nn.Softmax(dim=1),
  'Softmax2': nn.Softmax(dim=2)
}


def layer_init(layer, w_scale=1.0):
  # Initialize all weights and biases in layer and return it
  # nn.init.orthogonal_(layer.weight.data)
  # nn.init.xavier_normal_(layer.weight.data, gain=1)
  nn.init.kaiming_normal_(layer.weight.data, mode='fan_in', nonlinearity='relu')
  layer.weight.data.mul_(w_scale)
  nn.init.constant_(layer.bias.data, 0) # layer.bias.data.zero_()
  return layer


class MLP(nn.Module):
  '''
  Multilayer Perceptron
  '''
  def __init__(self, layer_dims, hidden_activation='ReLU', output_activation='None'):
    super().__init__()
    # Create layers
    self.mlp = nn.ModuleList([])
    for i in range(len(layer_dims[:-1])):
      dim_in, dim_out = layer_dims[i], layer_dims[i+1]
      self.mlp.append(layer_init(nn.Linear(dim_in, dim_out, bias=True)))
      if i+2 != len(layer_dims):
        if hidden_activation != 'None':
          self.mlp.append(activations[hidden_activation])
      elif output_activation != 'None':
        self.mlp.append(activations[output_activation])  
  
  def forward(self, x):
    for layer in self.mlp:
      x = layer(x)
    return x


class Conv2d_Atari(nn.Module):
  '''
  2D convolution neural network for Atari games
  '''
  def __init__(self, in_channels=4, feature_dim=512):
    super().__init__()
    self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
    self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
    self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
    linear_input_size = 7 * 7 * 64
    self.fc4 = layer_init(nn.Linear(linear_input_size, feature_dim))

  def forward(self, x):
    y = F.relu(self.conv1(x))
    y = F.relu(self.conv2(y))
    y = F.relu(self.conv3(y))
    y = y.view(y.size(0), -1)
    y = F.relu(self.fc4(y))
    return y


class Conv2d_MinAtar(nn.Module):
  '''
  2D convolution neural network for MinAtar games
  '''
  def __init__(self, in_channels, feature_dim=128):
    super().__init__()
    self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=1))
    def size_linear_unit(size, kernel_size=3, stride=1):
      return (size - (kernel_size - 1) - 1) // stride + 1
    linear_input_size = size_linear_unit(10) * size_linear_unit(10) * 16
    self.fc2 = layer_init(nn.Linear(linear_input_size, feature_dim))
    
  def forward(self, x):
    y = F.relu(self.conv1(x))
    y = y.view(y.size(0), -1)
    y = F.relu(self.fc2(y))
    return y


class NetworkGlue(nn.Module):
  '''
  Glue two networks
  '''
  def __init__(self, net1, net2):
    super().__init__()
    self.net1 = net1
    self.net2 = net2

  def forward(self, x):
    y = self.net2(self.net1(x))
    return y


class DQNNet(nn.Module):
  '''
  Glue the feature net with value net
    - feature net: generate feature given raw input
    - value net: output action value given feature input
  '''
  def __init__(self, feature_net, value_net):
    super().__init__()
    self.feature_net = feature_net
    self.value_net = value_net

  def forward(self, x):
    f = self.feature_net(x)
    v = self.value_net(f)
    return v


class CategoricalREINFORCENet(nn.Module):
  def __init__(self, feature_net, actor_net):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net

  def forward(self, x):
    # Get a probability distribution over action space
    action_probs = self.actor_net(self.feature_net(x))
    # Sample an action
    action_distribution = Categorical(probs=action_probs)
    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    return {'action': action, 'log_prob': log_prob}


class GaussianREINFORCENet(nn.Module):
  def __init__(self, feature_net, actor_net, action_size=-1):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    # The action std is independent of states
    self.action_std = nn.Parameter(torch.zeros(action_size))
    
  def forward(self, x):
    # Get the mean of probability distribution over action space
    action_mean = self.actor_net(self.feature_net(x)).squeeze(0)
    # Sample an action
    action_distribution = Normal(action_mean, F.softplus(self.action_std))
    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    return {'action': action, 'log_prob': log_prob}


class CategoricalActorCriticNet(nn.Module):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    self.critic_net = critic_net

  def forward(self, x):
    # Get a probability distribution over action space
    phi = self.feature_net(x)
    action_probs = self.actor_net(phi)
    state_value = self.critic_net(phi)
    # Sample an action
    action_distribution = Categorical(probs=action_probs)
    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    return {'action': action, 'log_prob': log_prob, 'state_value': state_value}

  
class GaussianActorCriticNet(nn.Module):
  def __init__(self, feature_net, actor_net, critic_net, action_size=-1):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    self.critic_net = critic_net
    # The action std is independent of states
    self.action_std = nn.Parameter(torch.zeros(action_size))
    
  def forward(self, x):
    # Get the mean of probability distribution over action space
    phi = self.feature_net(x)
    action_mean = self.actor_net(phi).squeeze(0)
    state_value = self.critic_net(phi).squeeze(0)
    # Sample an action
    action_distribution = Normal(action_mean, F.softplus(self.action_std))
    action = action_distribution.sample()
    log_prob = action_distribution.log_prob(action)
    return {'action': action, 'log_prob': log_prob, 'state_value': state_value}