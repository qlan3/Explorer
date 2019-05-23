import torch
import random


class BaseExploration(object):
  # Base class for agent exploration strategies. 
  def __init__(self, exploration):
    pass

  def select_action(self, q_values):
    raise NotImplementedError("To be implemented")


class EpsilonGreedy(BaseExploration):
  # Implementation of the epsilon greedy exploration strategy
  def __init__(self, exploration):
    super().__init__(exploration)
    self.epsilon = exploration['epsilon']

  def select_action(self, q_values):
    if random.random() > self.epsilon:
      action = torch.argmax(q_values).item()
    else:
      action = random.randint(0, q_values.shape[1] - 1)
    return action


class LinearEpsilonGreedy(BaseExploration):
  def __init__(self, exploration):
    super().__init__(exploration)
    self.inc = (exploration['end'] - exploration['start']) / float(exploration['steps'])
    self.epsilon = exploration['start']
    self.end = exploration['end']
    if exploration['end'] > exploration['end']:
      self.bound = min
    else:
      self.bound = max

  def select_action(self, q_values):
    if random.random() > self.epsilon:
      action = torch.argmax(q_values).item()
    else:
      action = random.randint(0, q_values.shape[1] - 1)
    self.epsilon = self.bound(self.epsilon + self.inc, self.end)
    return action