from utils.submitter import Submitter
from utils.helper import make_dir

def main():
  cfg = dict()
  # User name
  cfg['user'] = 'qlan3'
  # Project directory
  cfg['project_dir'] = '/home/qlan3/projects/def-afyshe-ab/qlan3/Explorer'
  # Sbatch script path
  cfg['script_path'] = './sbatch.sh'
  # Job indexes list
  cfg['job_list'] = list(range(1,28+1))
  # Check time interval in minutes
  cfg['check_time_interval'] = 0.01
  # cluster_name: cluster_capacity
  # cfg['clusters'] = {'Mp2':5000, 'Cedar':1000, 'Graham': 1000, 'Beluga':1000}
  cfg['clusters'] = {'Graham': 1000}

  make_dir('output/atari_ram_1')
  submitter = Submitter(cfg)
  submitter.submit()

if __name__=='__main__':
  main()