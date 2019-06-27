from utils.logger import *

class BaseAgent(object):
  def __init__(self, cfg):
    self.logger = Logger(cfg.logs_dir)