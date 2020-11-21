from agents.DQN import *


class DDQN(DQN):
  '''
  Implementation of Double DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def compute_q_target(self, batch):
    with torch.no_grad():
      best_actions = self.Q_net[0](batch.next_state).argmax(1).unsqueeze(1)
      q_next = self.Q_net_target[0](batch.next_state).gather(1, best_actions).squeeze()
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target