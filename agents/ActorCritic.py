from agents.REINFORCEWithBaseline import *


class ActorCritic(REINFORCEWithBaseline):
  '''
  Implementation of Actor-Critic (state value function)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def learn(self):
    # Compute advantage
    self.storage.placeholder(self.episode_step_count)
    self.storage.add({'v': torch.tensor(0.0)})
    for i in range(self.episode_step_count):
      self.storage.adv[i] = self.storage.reward[i] + self.discount * self.storage.mask[i] * self.storage.v[i+1].detach() - self.storage.v[i]
    # Get training data
    entries = self.storage.get(['log_prob', 'adv'], self.episode_step_count)
    # Compute loss
    actor_loss = -(entries.log_prob * entries.adv.detach()).mean()
    critic_loss = 0.5 * entries.adv.pow(2).mean()
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
    self.logger.debug(f'Step {self.step_count}: actor_loss={actor_loss.item()}, critic_loss={critic_loss.item()}')
    # Take an optimization step
    self.optimizer['actor'].zero_grad()
    self.optimizer['critic'].zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
      nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
    self.optimizer['actor'].step()
    self.optimizer['critic'].step()