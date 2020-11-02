import gym
import time
import copy
import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from utils.logger import *

class BaseAgent(object):
  def __init__(self, cfg):
    self.logger = Logger(cfg['logs_dir'])