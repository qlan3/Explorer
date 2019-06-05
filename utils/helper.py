import os
import sys
import torch
import psutil
import numpy as np
import datetime

def get_time_str():
  return datetime.datetime.now().strftime("%y.%m.%d-%H:%M:%S")

def rss_memory_usage():
  '''
  Return the resident memory usage in MB
  '''
  process = psutil.Process(os.getpid())
  mem = process.memory_info().rss / float(2 ** 20)
  return mem

def str_to_class(module_name, class_name):
  '''
  Convert string to class
  '''
  return getattr(sys.modules[module_name], class_name)

def set_one_thread():
  '''
  Set number of threads for pytorch to 1
  '''
  os.environ['OMP_NUM_THREADS'] = '1'
  os.environ['MKL_NUM_THREADS'] = '1'
  torch.set_num_threads(1)

def to_tensor(x, device):
  '''
  Convert an array to tensor
  '''
  if isinstance(x, torch.Tensor):
    return x
  x = np.asarray(x, dtype=np.float)
  x = torch.tensor(x, device=device, dtype=torch.float32)
  return x

def to_numpy(t):
  '''
  Convert a tensor to numpy
  '''
  if isinstance(t, torch.Tensor):
    return t.cpu().detach().numpy()
  else:
    return t