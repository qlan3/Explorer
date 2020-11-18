import math
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


class MLPCritic(nn.Module):
  def __init__(self, layer_dims, hidden_activation='ReLU', output_activation='None'):
    super().__init__()
    self.value_net = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation=output_activation)

  def forward(self, phi):
    return self.value_net(phi).squeeze(-1)


class MLPQCritic(nn.Module):
  def __init__(self, layer_dims, hidden_activation='ReLU', output_activation='None'):
    super().__init__()
    self.Q = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation=output_activation)

  def forward(self, phi, action):
    phi_action = torch.cat([phi, action], dim=-1)
    q = self.Q(phi_action).squeeze(-1)
    return q


class MLPDoubleQCritic(nn.Module):
  def __init__(self, layer_dims, hidden_activation='ReLU', output_activation='None'):
    super().__init__()
    self.Q1 = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation=output_activation)
    self.Q2 = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation=output_activation)

  def forward(self, phi, action):
    phi_action = torch.cat([phi, action], dim=-1)
    q1 = self.Q1(phi_action).squeeze(-1)
    q2 = self.Q2(phi_action).squeeze(-1)
    return q1, q2


class Actor(nn.Module):
  def distribution(self, phi):
    raise NotImplementedError

  def log_prob_from_distribution(self, action_distribution, action):
    raise NotImplementedError

  def forward(self, phi, action=None):
    # Compute action distribution and the log_prob of given actions
    action_distribution = self.distribution(phi)
    if action is None:
      action = action_distribution.sample()
    log_prob = self.log_prob_from_distribution(action_distribution, action)
    return action_distribution, action, log_prob


class MLPCategoricalActor(Actor):
  def __init__(self, layer_dims, hidden_activation='ReLU', output_activation='None'):
    super().__init__()
    self.logits_net = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation=output_activation)

  def distribution(self, phi):
    logits = self.logits_net(phi)
    return Categorical(logits=logits)

  def log_prob_from_distribution(self, action_distribution, action):
    return action_distribution.log_prob(action)


class MLPGaussianActor(Actor):
  def __init__(self, layer_dims, hidden_activation='ReLU', output_activation='None'):
    super().__init__()
    self.actor_net = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation=output_activation)
    # The action std is independent of states
    action_size = layer_dims[-1]
    self.action_std = nn.Parameter(torch.zeros(action_size))    

  def distribution(self, phi):
    action_mean = self.actor_net(phi)
    action_std = F.softplus(self.action_std)
    return Normal(action_mean, action_std)
    
  def log_prob_from_distribution(self, action_distribution, action):
    # Last axis sum needed for Torch Normal distribution
    return action_distribution.log_prob(action).sum(axis=-1)


class MLPSquashedGaussianActor(Actor):
  def __init__(self, action_lim, layer_dims, hidden_activation='ReLU', log_std_bounds=(-20, 2)):
    super().__init__()
    self.actor_net = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation='None')
    self.log_std_min, self.log_std_max = log_std_bounds
    self.action_lim = action_lim

  def distribution(self, phi):
    action_mean, action_log_std = self.actor_net(phi).chunk(2, dim=-1)
    # Constrain log_std inside [log_std_min, log_std_max]
    action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
    return action_mean, Normal(action_mean, action_log_std.exp())

  def log_prob_from_distribution(self, action_distribution, action):
    # NOTE: Check out the original SAC paper and https://github.com/openai/spinningup/issues/279 for details
    log_prob = action_distribution.log_prob(action).sum(axis=-1)
    log_prob -= (2*(math.log(2) - action - F.softplus(-2*action))).sum(axis=-1)
    return log_prob

  def forward(self, phi, action=None, deterministic=False):
    # Compute action distribution and the log_prob of given actions
    action_mean, action_distribution = self.distribution(phi)
    if deterministic:
      action = action_mean
    elif action is None:
      action = action_distribution.rsample()
    # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
    log_prob = self.log_prob_from_distribution(action_distribution, action)
    action = self.action_lim * torch.tanh(action)
    return action_distribution, action, log_prob


class MLPDeterministicActor(Actor):
  def __init__(self, action_lim, layer_dims, hidden_activation='ReLU'):
    super().__init__()
    self.actor_net = MLP(layer_dims=layer_dims, hidden_activation=hidden_activation, output_activation='Tanh')
    self.action_lim = action_lim
  
  def forward(self, phi):
    return self.action_lim * self.actor_net(phi)


class REINFORCENet(nn.Module):
  def __init__(self, feature_net, actor_net):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    self.actor_params = list(self.feature_net.parameters()) + list(self.actor_net.parameters())

  def forward(self, obs, action=None):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    _, action, log_prob = self.actor_net(phi, action)
    return {'action': action, 'log_prob': log_prob}


class ActorCriticNet(nn.Module):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    self.critic_net = critic_net
    self.actor_params = list(self.feature_net.parameters()) + list(self.actor_net.parameters())
    self.critic_params = list(self.feature_net.parameters()) + list(self.critic_net.parameters())

  def forward(self, obs, action=None):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute state value
    v = self.critic_net(phi)
    # Sample an action
    action_distribution, action, log_prob = self.actor_net(phi, action)
    return {'action': action, 'log_prob': log_prob, 'v': v}


class SACNet(ActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs, action=None, deterministic=False):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action_distribution, action, log_prob = self.actor_net(phi, action, deterministic)
    # Compute state-action value
    q1, q2 = self.critic_net(phi, action)
    return {'action': action, 'log_prob': log_prob, 'q1': q1, 'q2': q2}


class DeterministicActorCriticNet(ActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs, action=None):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    if action is None:
      action = self.actor_net(phi)
    # Compute state-action value
    q = self.critic_net(phi, action)
    return {'action': action, 'q': q}