import logging
from utils.helper import *
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
  def __init__(self, logs_dir, file_name='log.txt', filemode='w'):
    self.logs_dir = logs_dir
    logging.basicConfig(
      format='%(asctime)s - %(levelname)s: %(message)s',
      filename=f'{logs_dir}{file_name}',
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
    # Set default writer
    self.writer = None

  def init_writer(self):
    self.writer = SummaryWriter(self.logs_dir)
    self.add_scalar = self.writer.add_scalar       # Input: tag, scalar_value, global_step
    self.add_scalars = self.writer.add_scalars     # Input: main_tag, tag_scalar_dict, global_step
    self.add_histogram = self.writer.add_histogram # Input: tag, values, global_step