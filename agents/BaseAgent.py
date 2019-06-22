from utils.logger import *

class BaseAgent(object):
  def __init__(self, cfg):
    tb_logs_dir = cfg.logs_dir
    self.logger = Logger(tb_logs_dir)
    self.logger.debug(f'The tensorboard dir is: {tb_logs_dir}')
    # self.logger.init_writer()