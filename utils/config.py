class Config(object):
  def __init__(self):
    # Experiment parameters
    self.history_length = 4
    self.epsilon_decay = 0.999
    self.sgd_update_frequency = 1
    self.max_episode_steps = 0
    self.show_tb = False