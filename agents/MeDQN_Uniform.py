from agents.DQN import *


class MeDQN_Uniform(DQN):
  '''
  Implementation of MeDQN_Uniform (Memory-efficient DQN with uniform state sampling)
  - Consolidatie knowledge from target Q-network to current Q-network.
  - The bounds of state space are updated with real states frequently.
  - A tiny (e.g., one mini-batch size) experience replay buffer is used in practice.
  '''
  def __init__(self, cfg):
    # Set the consolidation batch size
    if 'consod_batch_size' not in cfg['agent'].keys():
      cfg['agent']['consod_batch_size'] = cfg['batch_size']
    super().__init__(cfg)
    self.replay = getattr(components.replay, cfg['memory_type'])(cfg['memory_size'], keys=['state', 'action', 'next_state', 'reward', 'mask'])
    # Set uniform state sampler for knowledge consolidation
    if 'MinAtar' in self.env_name:
      self.state_sampler = DiscreteUniformSampler(
        shape=self.env['Train'].observation_space.shape,
        normalizer=self.state_normalizer,
        device=self.device
      )
    else:
      self.state_sampler = ContinousUniformSampler(
        shape=self.env['Train'].observation_space.shape,
        normalizer=self.state_normalizer,
        device=self.device
      )
    # Set consolidation regularization strategy
    epsilon = {
      'steps': float(cfg['train_steps']),
      'start': cfg['agent']['consod_start'],
      'end': cfg['agent']['consod_end']
    }
    self.consolidate = getattr(components.exploration, 'LinearEpsilonGreedy')(-1, epsilon)
  
  def save_experience(self):
    super().save_experience()
    self.state_sampler.update_bound(self.original_state)

  def learn(self):
    mode = 'Train'
    batch = self.replay.get(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['memory_size'])
    q_target = self.compute_q_target(batch)
    lamda = self.consolidate.get_epsilon(self.step_count) # Compute consolidation regularization parameter
    for _ in range(self.cfg['agent']['consod_epoch']):
      q = self.compute_q(batch)
      sample_state = self.state_sampler.sample(self.cfg['agent']['consod_batch_size'])
      # Compute loss
      loss = self.loss(q, q_target)
      loss += lamda * self.consolidation_loss(sample_state)
      # Take an optimization step
      self.optimizer[0].zero_grad()
      loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.Q_net[0].parameters(), self.gradient_clip)
      self.optimizer[0].step()
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

  def consolidation_loss(self, state):
    q_values = self.Q_net[0](state).squeeze()
    q_target_values = self.Q_net_target[0](state).squeeze().detach()
    loss = nn.MSELoss(reduction='mean')(q_values, q_target_values)
    return loss