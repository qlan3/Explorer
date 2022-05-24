from agents.DQN import *
from collections import Counter


class BootstrappedDQN(DQN):
  '''
  Implementation of Bootstrapped DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    self.k = cfg['agent']['target_networks_num'] # number of head networks
    super().__init__(cfg)

  def createNN(self, input_type):
    # Set feature network
    if input_type == 'pixel':
      layer_dims = [self.cfg['feature_dim']] + self.cfg['hidden_layers'] + [self.action_size]
      if 'MinAtar' in self.env_name:
        feature_net = Conv2d_MinAtar(in_channels=self.env['Train'].game.state_shape()[2], feature_dim=layer_dims[0])
      else:
        feature_net = Conv2d_Atari(in_channels=4, feature_dim=layer_dims[0])
    elif input_type == 'feature':
      layer_dims = [self.state_size] +self.cfg['hidden_layers'] + [self.action_size]
      feature_net = nn.Identity()
    # Set value network
    assert self.action_type == 'DISCRETE', f'{self.agent_name} only supports discrete action spaces.'    
    heads_net = nn.ModuleList([
      MLPCritic(layer_dims=layer_dims, hidden_act=self.cfg['hidden_act'], output_act=self.cfg['output_act'], last_w_scale=1.0)
      for _ in range(self.k)
    ])
    NN = BootstrappedDQNNet(feature_net, heads_net)
    return NN

  def get_action(self, mode='Train'):
    '''
    Uses the local Q network to pick an action
    PyTorch only accepts mini-batches and not single observations so we have to use unsqueeze to add
    a "fake" dimension to make it a mini-batch rather than a single observation
    '''
    state = to_tensor(self.state[mode], device=self.device)
    state = state.unsqueeze(0) # Add a batch dimension (Batch, Channel, Height, Width)
    if mode == 'Test':
      q_values_all = self.Q_net[0](state)
      actions = [np.argmax(to_numpy(q_values).flatten()) for q_values in q_values_all]
      action, _ = Counter(actions).most_common()[0]
    elif mode == 'Train':
      q_values = self.get_action_selection_q_values(state)
      action = np.argmax(q_values)
    return action

  def learn(self):
    mode = 'Train'
    batch = self.replay.sample(['state', 'action', 'reward', 'next_state', 'mask'], self.cfg['batch_size'])
    qs, q_targets = self.compute_q(batch), self.compute_q_target(batch)
    # Compute loss
    loss = 0
    for i in range(self.k):
      loss += self.loss(qs[i], q_targets[i])
    loss /= self.k
    # Take an optimization step
    self.optimizer[0].zero_grad()
    loss.backward()
    if self.gradient_clip > 0:
      nn.utils.clip_grad_norm_(self.Q_net[0].parameters(), self.gradient_clip)
    self.optimizer[0].step()
    if self.show_tb:
      self.logger.add_scalar(f'Loss', loss.item(), self.step_count)  
  
  def compute_q_target(self, batch):
    q_targets = []
    with torch.no_grad():
      q_nexts = self.Q_net_target[0](batch.next_state)
      for i in range(self.k):
        q_next = q_nexts[i].max(1)[0]
        q_target = batch.reward + self.discount * q_next * batch.mask
        q_targets.append(q_target)
    return q_targets
    
  def compute_q(self, batch):
    # Convert actions to long so they can be used as indexes
    action = batch.action.long().unsqueeze(1)
    q_outputs = self.Q_net[0](batch.state)
    qs = []
    for i in range(self.k):
      q = q_outputs[i].gather(1, action).squeeze()
      qs.append(q)
    return qs

  def get_action_selection_q_values(self, state):
    head_idx = random.randrange(self.k)
    q_values = self.Q_net[0](state, head_idx)
    q_values = to_numpy(q_values).flatten()
    return q_values