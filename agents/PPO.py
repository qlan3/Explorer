from agents.ActorCritic import *


class PPO(ActorCritic):
  '''
  Implementation of PPO (Proximal Policy Optimization)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set replay buffer
    self.replay = FiniteReplay(self.cfg['steps_per_epoch']+1, keys=['state', 'action', 'reward', 'mask', 'v', 'log_pi', 'ret', 'adv'])

  def save_experience(self, prediction):
    # Save state, action, reward, mask, v, log_pi
    mode = 'Train'
    if self.reward[mode] is not None:
      prediction = {
        'state': to_tensor(self.state[mode], self.device),
        'action': to_tensor(self.action[mode], self.device),
        'reward': to_tensor(self.reward[mode], self.device),
        'mask': to_tensor(1-self.done[mode], self.device),
        'v': prediction['v'],
        'log_pi': prediction['log_pi']
      }
      self.replay.add(prediction)
    else:
      self.replay.add({'v': prediction['v']})

  def learn(self):
    mode = 'Train'
    # Compute return and advantage
    adv = torch.tensor(0.0)
    ret = self.replay.v[-1].detach()
    for i in reversed(range(self.cfg['steps_per_epoch'])):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      if self.cfg['gae'] < 0:
        adv = ret - self.replay.v[i].detach()
      else:
        td_error = self.replay.reward[i] + self.discount * self.replay.mask[i] * self.replay.v[i+1] - self.replay.v[i]
        adv = self.discount * self.cfg['gae'] * self.replay.mask[i] * adv + td_error
      self.replay.adv[i] = adv.detach()
      self.replay.ret[i] = ret.detach()
    # Get training data and **detach** (IMPORTANT: we don't optimize old parameters)
    entries = self.replay.get(['log_pi', 'ret', 'adv', 'state', 'action'], self.cfg['steps_per_epoch'], detach=True)
    # Normalize advantages
    entries.adv.copy_((entries.adv - entries.adv.mean()) / entries.adv.std())
    # Optimize for multiple epochs
    for _ in range(self.cfg['optimize_epochs']):
      batch_idxs = generate_batch_idxs(len(entries.log_pi), self.cfg['batch_size'])
      for batch_idx in batch_idxs:
        batch_idx = to_tensor(batch_idx, self.device).long()
        prediction = self.network(entries.state[batch_idx], entries.action[batch_idx])
        # Take an optimization step for actor
        approx_kl = (entries.log_pi[batch_idx] - prediction['log_pi']).mean()
        if approx_kl <= 1.5 * self.cfg['target_kl']:
          ratio = torch.exp(prediction['log_pi'] - entries.log_pi[batch_idx])
          obj = ratio * entries.adv[batch_idx]
          obj_clipped = torch.clamp(ratio, 1-self.cfg['clip_ratio'], 1+self.cfg['clip_ratio']) * entries.adv[batch_idx]
          actor_loss = -torch.min(obj, obj_clipped).mean()
          self.optimizer['actor'].zero_grad()
          actor_loss.backward()
          if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
          self.optimizer['actor'].step()
        # Take an optimization step for critic
        critic_loss = (entries.ret[batch_idx] - prediction['v']).pow(2).mean()
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.gradient_clip > 0:
          nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
        self.optimizer['critic'].step()
    # Log
    if self.show_tb:
      self.logger.add_scalar(f'actor_loss', actor_loss.item(), self.step_count)
      self.logger.add_scalar(f'critic_loss', critic_loss.item(), self.step_count)
      self.logger.add_scalar(f'log_pi', entries.log_pi.mean().item(), self.step_count)