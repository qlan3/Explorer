from agents.VanillaDQN import *


class AveragedDQN(VanillaDQN):
  '''
  Implementation of Averaged DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.k = cfg['agent']['target_networks_num'] # number of target networks
    # Create target Q value network
    self.Q_net_target = [None] * self.k
    for i in range(self.k):
      self.Q_net_target[i] = self.createNN(cfg['env']['input_type']).to(self.device)
      # Load target Q value network
      self.Q_net_target[i].load_state_dict(self.Q_net[0].state_dict())
      self.Q_net_target[i].eval()
    self.update_target_net_index = 0
  
  def learn(self):
    super().learn()
    # Update target network
    if (self.step_count // self.cfg['network_update_frequency']) % self.cfg['target_network_update_frequency'] == 0:
      self.Q_net_target[self.update_target_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())
      self.update_target_net_index = (self.update_target_net_index + 1) % self.k
 
  def compute_q_target(self, batch):
    with torch.no_grad():
      q_sum = self.Q_net_target[0](batch.next_state).clone()
      for i in range(1, self.k):
        q = self.Q_net_target[i](batch.next_state)
        q_sum = q_sum + q
      q_next = q_sum.max(1)[0] / self.k
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target