import torch

from agents.MaxminDQN import MaxminDQN, MaxminDQN_v1


class AveragedDQN(MaxminDQN_v1):
  '''
  Implementation of Averaged DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)
 
  def compute_q_target(self, next_states, rewards, dones):
    q_average = self.Q_net(next_states).clone().detach()
    for i in range(self.k):
      q = self.Q_net_target[i](next_states).detach()
      q_average = q_average + q
    q_next = q_average.max(1)[0] / (self.k+1)
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target