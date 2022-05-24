from agents.VanillaDQN import *


class DQN(VanillaDQN):
  '''
  Implementation of DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
    # Create target Q value network
    self.Q_net_target = [None]
    self.Q_net_target[0] = self.createNN(cfg['env']['input_type']).to(self.device)
    # Load target Q value network
    self.Q_net_target[0].load_state_dict(self.Q_net[0].state_dict())
    self.Q_net_target[0].eval()

  def update_target_net(self):
    if self.step_count % self.cfg['target_network_update_steps'] == 0:
      self.Q_net_target[self.update_Q_net_index].load_state_dict(self.Q_net[self.update_Q_net_index].state_dict())

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_next = self.Q_net_target[0](batch.next_state).max(1)[0]
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target