from agents.VanillaDQN import VanillaDQN


class DQN(VanillaDQN):
  '''
  Implementation of DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    self.target_network_update_freqency = cfg['target_network_update_freqency']
    # Create target Q value network
    self.Q_net_target = [None]
    self.Q_net_target[0] = self.creatNN(cfg['env']['input_type']).to(self.device)
    # Load target Q value network
    self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
    self.Q_net_target[0].eval()

  def learn(self):
    super().learn()
    # Update target network
    if (self.step_count // self.sgd_update_frequency) % self.target_network_update_freqency == 0:
      self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

  def compute_q_target(self, next_states, rewards, dones):
    q_next = self.Q_net_target[0](next_states).detach().max(1)[0]
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target