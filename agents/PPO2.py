from agents.PPO import *


class PPO2(PPO):
  '''
  Implementation of PPO (Proximal Policy Optimization): spinning up version, better performance, but much slower
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

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
      # Optimize the actor
      for i in range(self.cfg['train_iters']):
        prediction = self.network(entries.state, entries.action)
        approx_kl = (entries.log_pi - prediction['log_pi']).mean().item()
        if approx_kl > 1.5 * self.cfg['target_kl']:
          break
        # Compute actor loss
        ratio = torch.exp(prediction['log_pi'] - entries.log_pi)
        obj = ratio * entries.adv
        obj_clipped = torch.clamp(ratio, 1-self.cfg['clip_ratio'], 1+self.cfg['clip_ratio']) * entries.adv
        actor_loss = -torch.min(obj, obj_clipped).mean()
        self.optimizer['actor'].zero_grad()
        actor_loss.backward()
        if self.gradient_clip > 0:
          nn.utils.clip_grad_norm_(self.network.actor_params, self.gradient_clip)
        self.optimizer['actor'].step()
      # Optimize the critic
      for i in range(self.cfg['train_iters']):
        prediction = self.network(entries.state, entries.action)
        critic_loss = (entries.ret - prediction['v']).pow(2).mean()
        self.optimizer['critic'].zero_grad()
        critic_loss.backward()
        if self.gradient_clip > 0:
          nn.utils.clip_grad_norm_(self.network.critic_params, self.gradient_clip)
        self.optimizer['critic'].step()