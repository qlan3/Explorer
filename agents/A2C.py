import torch
from torch import nn

from utils.helper import *
from components.network import *
from agents.REINFORCEWithBaseline import REINFORCEWithBaseline


class A2C(REINFORCEWithBaseline):
  '''
  Implementation of Actor-Critic
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def learn(self):
    # Compute target state value
    rewards = to_tensor(self.episode['reward'], device=self.device)
    dones = to_tensor(self.episode['done'], device=self.device)
    state_values = torch.cat(self.episode['state_value']).squeeze()
    # Set the state value of the end state in an episode to be 0
    target_values = torch.cat((state_values.clone()[1:], torch.tensor([0.0]))).detach()
    target_values = rewards + (1-dones) * self.discount * target_values
    # Compute adavantage
    advantages = (target_values - state_values).detach()
    # Compute actor loss
    actor_loss = []
    for log_prob, advantage in zip(self.episode['log_prob'], advantages):
      actor_loss.append(-log_prob * advantage)
    actor_loss = torch.cat(actor_loss).sum()
    # Compute critic loss
    critic_loss = 0.5 * (target_values - state_values).pow(2).sum()
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