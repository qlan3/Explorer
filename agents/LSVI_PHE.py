from agents.MaxminDQN import *


class LSVI_PHE(MaxminDQN):
  '''
  Implementation of LSVI_PHE with target network and replay buffer.
  We update all Q_nets for every update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    self.noise_std = cfg['agent']['noise_std'] # std of Gaussian noise
    self.l2_lambda = cfg['agent']['lambda'] # noise regularization parameter

  def learn(self):
    mode = 'Train'
    # Use the same batch during training
    batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    # Update Q network
    for i in range(self.k): 
      self.update_Q_net_index = i
      q, q_target = self.compute_q(batch), self.compute_q_target(batch)
      # Compute L2 noise regularization
      l2_reg = torch.tensor(0.0)
      for param in self.Q_net[i].parameters():
        param_noise = to_tensor(torch.randn(param.size())*self.noise_std, device=self.device)
        l2_reg += torch.linalg.norm(param + param_noise)
      # Compute loss
      loss = self.loss(q, q_target) + self.l2_lambda * l2_reg
      # Take an optimization step
      self.optimizer[i].zero_grad()
      loss.backward()
      if self.gradient_clip > 0:
        nn.utils.clip_grad_norm_(self.Q_net[i].parameters(), self.gradient_clip)
      self.optimizer[i].step()
    # Update target network
    if (self.step_count // self.cfg['network_update_steps']) % self.cfg['target_network_update_steps'] == 0:
      for i in range(self.k):
        self.Q_net_target[i].load_state_dict(self.Q_net[i].state_dict())
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    q_values = self.get_action_selection_q_values(state)
    # Alwaya select the best action
    action = np.argmax(q_values)
    return action

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_max = self.Q_net_target[0](batch.next_state).clone()
      for i in range(1, self.k):
        q = self.Q_net_target[i](batch.next_state)
        q_max = torch.max(q_max, q)
      q_next = q_max.max(1)[0]
      reward_noise = to_tensor(torch.randn(batch.reward.size())*self.noise_std, device=self.device)
      q_target = batch.reward + reward_noise + self.discount * q_next * batch.mask
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_max = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_max = torch.max(q_max, q)
    q_max = to_numpy(q_max).flatten()
    return q_max