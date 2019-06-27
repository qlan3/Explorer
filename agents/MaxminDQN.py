import torch

from agents.VanillaDQN import VanillaDQN


class MaxminDQN(VanillaDQN):
  '''
  Implementation of Maxmin DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.target_network_update_freqency = cfg.target_network_update_freqency
    self.k = cfg.target_networks_num # number of target networks
    # Create target Q value network
    self.Q_net_target = [None] * self.k
    for i in range(self.k):
      self.Q_net_target[i] = self.creatNN(cfg.input_type).to(self.device)
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net.state_dict())
      self.Q_net_target[i].eval()
    self.update_target_net_index = 0

  def learn(self):
    super().learn()
    # Update target network
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      self.Q_net_target[self.update_target_net_index].load_state_dict(self.Q_net.state_dict())
      self.update_target_net_index = (self.update_target_net_index + 1) % self.k
 
  def compute_q_target(self, next_states, rewards, dones):
    q_min = self.Q_net(next_states).clone().detach()
    for i in range(self.k):
      q = self.Q_net_target[i](next_states).detach()
      q_min = torch.min(q_min, q)
    q_next = q_min.max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target