import torch
import numpy as np
from collections import namedtuple
from utils.helper import to_tensor


class InfiniteReplay(object):
  '''
  Infinite replay buffer to store experiences
  '''
  def __init__(self, keys=None):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.clear()

  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k).append(v)

  def placeholder(self, data_size):
    for k in self.keys:
      v = getattr(self, k)
      if len(v) == 0:
        setattr(self, k, [None] * data_size)

  def clear(self):
    for key in self.keys:
      setattr(self, key, [])

  def get(self, keys, data_size):
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))


class FiniteReplay(object):
  '''
  Finite replay buffer to store experiences: FIFO (first in, firt out)
  '''
  def __init__(self, memory_size, keys=None):
    if keys is None:
      keys = ['action', 'reward', 'mask']
    self.keys = keys
    self.memory_size = int(memory_size)
    self.clear()

  def clear(self):
    self.pos = 0
    self.full = False
    for key in self.keys:
      setattr(self, key, [None] * self.memory_size)

  def add(self, data):
    for k, v in data.items():
      if k not in self.keys:
        raise RuntimeError('Undefined key')
      getattr(self, k)[self.pos] = v
    self.pos = (self.pos + 1) % self.memory_size
    if self.pos == 0:
      self.full = True

  def get(self, keys, data_size, detach=False):
    # Get first several samples (without replacement)
    data_size = min(self.size(), data_size) 
    data = [getattr(self, k)[:data_size] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def sample(self, keys, batch_size, detach=False):
    # Sampling with replacement
    idxs = np.random.randint(0, self.size(), size=batch_size)
    data = [[getattr(self, k)[idx] for idx in idxs] for k in keys]
    data = map(lambda x: torch.stack(x), data)
    if detach:
      data = map(lambda x: x.detach(), data)
    Entry = namedtuple('Entry', keys)
    return Entry(*list(data))

  def is_empty(self):
    if self.pos == 0 and not self.full:
      return True
    else:
      return False
  
  def is_full(self):
    return self.full

  def size(self):
    if self.full:
      return self.memory_size
    else:
      return self.pos


class ContinousUniformSampler(object):
  '''
  A uniform sampler for continous space
  '''
  def __init__(self, shape, normalizer, device):
    self.shape = shape
    self.normalizer = normalizer
    self.device = device
    self.reset()
  
  def reset(self):
    self.low = np.inf * np.ones(self.shape)
    self.high = -np.inf * np.ones(self.shape)
  
  def update_bound(self, data):
    self.low = np.minimum(self.low, data)
    self.high = np.maximum(self.high, data)

  def sample(self, batch_size):
    data = np.random.uniform(low=self.low, high=self.high, size=tuple([batch_size]+list(self.shape)))
    data = to_tensor(self.normalizer(data), self.device)
    return data


class DiscreteUniformSampler(ContinousUniformSampler):
  '''
  A uniform sampler for discrete space
  '''
  def __init__(self, shape, normalizer, device):
    super().__init__(shape, normalizer, device)

  def sample(self, batch_size):
    data = np.random.randint(low=np.rint(self.low), high=np.rint(self.high)+1, size=tuple([batch_size]+list(self.shape)))
    data = to_tensor(self.normalizer(data), self.device)
    return data