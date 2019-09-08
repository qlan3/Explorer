from agents.VanillaDQN import VanillaDQN


class AveragedDQN(VanillaDQN):
  '''
  Implementation of Averaged DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.target_network_update_freqency = cfg['target_network_update_freqency']
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    # Create target Q value network
    self.Q_net_target = [None] * self.k
    for i in range(self.k):
      self.Q_net_target[i] = self.creatNN(cfg['env']['input_type']).to(self.device)
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[0].state_dict())
      self.Q_net_target[i].eval()
    self.update_target_net_index = 0
  
  def learn(self):
    super().learn()
    # Update target network
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      self.Q_net_target[self.update_target_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())
      self.update_target_net_index = (self.update_target_net_index + 1) % self.k
 
  def compute_q_target(self, next_states, rewards, dones):
    q_sum = self.Q_net_target[0](next_states).clone().detach()
    for i in range(1, self.k):
      q = self.Q_net_target[i](next_states).detach()
      q_sum = q_sum + q
    q_next = q_sum.max(1)[0] / self.k
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target