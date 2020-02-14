import logging
from utils.helper import *
from tensorboardX import SummaryWriter

class Logger(object):
  def __init__(self, logs_dir, file_name='log.txt', filemode='w'):
    logging.basicConfig(
      format='%(asctime)s - %(levelname)s: %(message)s',
      filename=f'{logs_dir}/{file_name}',
      filemode=filemode
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    self.debug = logger.debug
    self.info = logger.info
    self.warning = logger.warning
    self.error = logger.error
    self.critical = logger.critical
    
    self.logs_dir = logs_dir
    self.writer = None

  def init_writer(self):
    self.writer = SummaryWriter(self.logs_dir)

  def add_scalar(self, tag, scalar_value, global_step=None):
    self.writer.add_scalar(tag, scalar_value, global_step)

  def add_scalars(self, main_tag, tag_scalar_dict, global_step=None):
    self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

  def add_histogram(self, tag, values, global_step=None):
    self.writer.add_histogram(tag, values, global_step)