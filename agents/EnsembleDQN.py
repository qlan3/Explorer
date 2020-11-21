from agents.MaxminDQN import *


class EnsembleDQN(MaxminDQN):
  '''
  Implementation of Ensemble DQN with target network and replay buffer
  
  In the original paper, all Q_nets are updated in Ensemble DQN for every update.
  However, this makes training really slow. Instead, we randomly choose one to update.
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def compute_q_target(self, batch):
    with torch.no_grad():
      q_ensemble = self.Q_net_target[0](batch.next_state).clone()
      for i in range(1, self.k):
        q = self.Q_net_target[i](batch.next_state)
        q_ensemble = q_ensemble + q
      q_next = q_ensemble.max(1)[0] / self.k
      q_target = batch.reward + self.discount * q_next * batch.mask
    return q_target
  
  def get_action_selection_q_values(self, state):
    q_ensemble = self.Q_net[0](state)
    for i in range(1, self.k):
      q = self.Q_net[i](state)
      q_ensemble = q_ensemble + q
    q_ensemble = to_numpy(q_ensemble / self.k).flatten()
    return q_ensemble