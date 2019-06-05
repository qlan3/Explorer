from utils.logger import *

class BaseAgent(object):
  def __init__(self, cfg, run):
    self.run = run
    tb_log_dir = cfg.log_dir
    self.logger = Logger(tb_log_dir)
    self.logger.debug(f'The tensorboard dir is: {tb_log_dir}')
    # self.logger.init_writer()