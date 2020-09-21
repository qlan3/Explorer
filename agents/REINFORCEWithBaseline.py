import torch
from torch import nn

from utils.helper import *
from components.network import *
from agents.REINFORCE import REINFORCE


class REINFORCEWithBaseline(REINFORCE):
  '''
  Implementation of REINFORCE with baseline
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    cfg.setdefault('critic_loss_weight', 1e-3)
    self.critic_loss_weight = cfg['critic_loss_weight']

  
  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      self.layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers']
      if 'MinAtar' in self.env_name:
        feature_net_head = Conv2d_MinAtar(in_channels=self.history_length, feature_dim=self.layer_dims[0])
      else:
        feature_net_head = Conv2d_Atari(in_channels=self.history_length, feature_dim=self.layer_dims[0])
      feature_net_body = MLP(layer_dims=self.layer_dims, hidden_activation=self.hidden_activation, output_activation=self.hidden_activation)
      feature_net = NetworkGlue(feature_net_head, feature_net_body)
    elif input_type == 'feature':
      self.layer_dims = [self.state_size] + self.cfg['hidden_layers']
      feature_net = MLP(layer_dims=self.layer_dims, hidden_activation=self.hidden_activation, output_activation=self.hidden_activation)
    # Set actor network and model
    if self.action_type == 'DISCRETE':
      actor_net = MLP(layer_dims=[self.layer_dims[-1], self.action_size], hidden_activation=self.hidden_activation, output_activation='Softmax-1')
      critic_net = MLP(layer_dims=[self.layer_dims[-1], 1], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
      NN = CategoricalActorCriticNet(feature_net, actor_net, critic_net)
    elif self.action_type == 'CONTINUOUS':
      actor_net = MLP(layer_dims=[self.layer_dims[-1], self.action_size], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
      critic_net = MLP(layer_dims=[self.layer_dims[-1], 1], hidden_activation=self.hidden_activation, output_activation=self.output_activation)
      NN = GaussianActorCriticNet(feature_net, actor_net, critic_net, self.action_size)
      
    return NN


  def reset_game(self):
    # Reset the game before a new episode
    super().reset_game()
    self.episode['state_value'] = []


  def learn(self):
    discounted_returns = to_tensor(self.episode['reward'], device=self.device)
    dones = to_tensor(self.episode['done'], device=self.device)
    state_values = torch.cat(self.episode['state_value']).squeeze()
    # Calculate the cumulative discounted return for an episode
    for i in range(len(discounted_returns)-2, -1, -1):
      discounted_returns[i] = discounted_returns[i] + (1-dones[i]) * self.discount * discounted_returns[i+1]
    # Compute adavantage
    advantages = (discounted_returns - state_values).detach()
    # Compute actor loss
    actor_loss = []
    for log_prob, advantage in zip(self.episode['log_prob'], advantages):
      actor_loss.append(-log_prob * advantage)
    actor_loss = torch.cat(actor_loss).sum()
    # Compute critic loss
    critic_loss = 0.5 * (discounted_returns - state_values).pow(2).sum()
    # Total loss
    loss = actor_loss + self.critic_loss_weight * critic_loss

    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: loss={loss.item()}')
    # Take an optimization step
    self.optimizer.zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clip)
    self.optimizer.step()

  def save_experience(self, prediction):
    # Save reward and log_prob for loss computation
    super().save_experience(prediction)
    self.episode['state_value'].append(prediction['state_value'])