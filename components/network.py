import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, w_scale=1.0):
  # Initialize all weights and biases in layer and return it
  nn.init.xavier_normal_(layer.weight.data, gain=1)
  layer.weight.data.mul_(w_scale)
  nn.init.constant_(layer.bias.data, 0) # layer.bias.data.zero_()
  return layer

class NetworkGlue(nn.Module):
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


class MLP(nn.Module):
  # Multilayer Perceptron
  def __init__(self, layer_dims, hidden_activation=nn.ReLU(), output_activation=None):
    super().__init__()
    # Create layers
    self.mlp = nn.ModuleList([])
    for i in range(len(layer_dims[:-1])):
      dim_in, dim_out = layer_dims[i], layer_dims[i+1]
      self.mlp.append(layer_init(nn.Linear(dim_in, dim_out, bias=True)))
      if i+2 != len(layer_dims) or output_activation == None:
        self.mlp.append(hidden_activation)
      else:
        self.mlp.append(output_activation)  
  
  def forward(self, x):
    for layer in self.mlp:
      x = layer(x)
    return x


class Conv2d_NN(nn.Module):
  # 2D convolution neural network
  def __init__(self, in_channels=4, h=84, w=84, feature_dim=512):
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