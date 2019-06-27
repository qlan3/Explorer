import torch
import numpy as np
from collections import namedtuple, deque
from utils.helper import to_tensor

class Replay(object):
  '''
  Replay buffer to store experiences
  '''
  def __init__(self, memory_size, batch_size, device):
    self.memory_size = int(memory_size)
    self.batch_size = batch_size
    self.device = device
    self.experience = namedtuple("Experience", field_names=["state", "action", "next_state", "reward", "done"])
    self.memory = []
    self.pos = 0

  def add(self, experience):
    '''
    Add experience(s) into memory
    '''
    for exp in experience:
      if len(self.memory) < self.memory_size:
        self.memory.append(None)
      self.memory[self.pos] = self.experience(*exp)
      self.pos = (self.pos + 1) % self.memory_size
    
  def sample(self):
    if self.empty():
      return None
    sampled_indices = [np.random.randint(0, len(self.memory)) for _ in range(self.batch_size)]
    sampled_data = [self.memory[idx] for idx in sampled_indices]
    # Stack sampled data and convert them to tensors
    states = to_tensor([np.asarray(e.state) for e in sampled_data if e is not None], self.device)
    actions = to_tensor([np.asarray([e.action]) for e in sampled_data if e is not None], self.device)
    next_states = to_tensor([np.asarray(e.next_state) for e in sampled_data if e is not None], self.device)
    rewards = to_tensor([e.reward for e in sampled_data if e is not None], self.device)
    dones = to_tensor([int(e.done) for e in sampled_data if e is not None], self.device)
    sampled_data = [states, actions, next_states, rewards, dones]
    return sampled_data

  def __len__(self):
    return len(self.memory)

  def size(self):
    return self.memory_size

  def empty(self):
    return len(self.memory) == 0

  def shuffle(self):
    np.random.shuffle(self.memory)