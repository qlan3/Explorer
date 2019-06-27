import logging
from utils.helper import *
from tensorboardX import SummaryWriter

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s')
class Logger(object):
  def __init__(self, tb_logs_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    self.debug = logger.debug
    self.info = logger.info
    self.warning = logger.warning
    self.error = logger.error
    self.critical = logger.critical
    
    self.tb_logs_dir = tb_logs_dir
    self.writer = None

  def init_writer(self):
    self.writer = SummaryWriter(self.tb_logs_dir)

  def add_scalar(self, tag, scalar_value, global_step=None):
    self.writer.add_scalar(tag, scalar_value, global_step)

  def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
    self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

  def add_histogram(self, tag, values, global_step=None):
    self.writer.add_histogram(tag, values, global_step)