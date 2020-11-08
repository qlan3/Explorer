from agents.A2C import *


class PPO(A2C):
  '''
  Implementation of PPO (Proximal Policy Optimization)
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Set replay buffer
    self.replay = FiniteReplay(self.steps_per_epoch+1, keys=['state', 'action', 'reward', 'mask', 'v', 'log_prob', 'ret', 'adv'])

  def save_experience(self, prediction):
    if self.reward is not None:
      prediction['mask'] = to_tensor(1-self.done, self.device)
      prediction['reward'] = to_tensor(self.reward, self.device)
      prediction['state'] = to_tensor(self.state, self.device)
    self.replay.add(prediction)

  def learn(self):
    # Compute return and advantage
    adv = torch.tensor(0.0)
    ret = self.replay.v[-1].detach()
    for i in reversed(range(self.steps_per_epoch)):
      ret = self.replay.reward[i] + self.discount * self.replay.mask[i] * ret
      if self.cfg['gae'] < 0:
        adv = ret - self.replay.v[i].detach()
      else:
        td_error = self.replay.reward[i] + self.discount * self.replay.mask[i] * self.replay.v[i+1] - self.replay.v[i]
        adv = self.discount * self.cfg['gae'] * self.replay.mask[i] * adv + td_error
      self.replay.adv[i] = adv.detach()
      self.replay.ret[i] = ret.detach()
    # Get training data and **detach** (IMPORTANT: we don't optimize old parameters)
    entries = self.replay.get(['log_prob', 'ret', 'adv', 'state', 'action'], self.steps_per_epoch)
    EntryCLS = entries.__class__
    entries = EntryCLS(*list(map(lambda x: x.detach(), entries)))
    # Normalize advantages
    entries.adv.copy_((entries.adv - entries.adv.mean()) / entries.adv.std())
    # Optimize for multiple epochs
    for t in range(self.cfg['optimize_epochs']):
      batch_idxs = generate_batch_idxs(len(entries.log_prob), self.cfg['batch_size'])
      for batch_idx in batch_idxs:
        batch_idx = to_tensor(batch_idx, self.device).long()
        # Compute losses
        prediction = self.network(entries.state[batch_idx], entries.action[batch_idx])
        ratio = torch.exp(prediction['log_prob'] - entries.log_prob[batch_idx])
        obj = ratio * entries.adv[batch_idx]
        obj_clipped = torch.clamp(ratio, 1-self.cfg['clip_ratio'], 1+self.cfg['clip_ratio']) * entries.adv[batch_idx]
        actor_loss = -torch.min(obj, obj_clipped).mean()
        critic_loss = 0.5 * (entries.ret[batch_idx] - prediction['v']).pow(2).mean()
        # Take an optimization step for actor
        approx_kl = (entries.log_prob[batch_idx] - prediction['log_prob']).mean().item()
        if approx_kl <= 1.5 * self.cfg['target_kl']:
          self.optimizer['actor'].zero_grad()
          actor_loss.backward()
          if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
          self.optimizer['actor'].step()
        # Take an optimization step for critic
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.gradient_clip > 0:
          nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
        self.optimizer['critic'].step()