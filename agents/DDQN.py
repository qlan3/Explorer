from agents.DQN import DQN


class DDQN(DQN):
  '''
  Implementation of Double DQN with target network and replay buffer
  '''
  def __init__(self, cfg):
    super().__init__(cfg)

  def compute_q_target(self, next_states, rewards, dones):
    best_actions = self.Q_net[0](next_states).detach().argmax(1).unsqueeze(1)
    q_next = self.Q_net_target[0](next_states).detach().gather(1, best_actions).squeeze()
    q_target = rewards + self.discount * q_next * (1 - dones)
    return q_target