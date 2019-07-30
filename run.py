import os
from utils.submitter import Submitter
#from utils.helper import make_dir

def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)

def main():
  cfg = dict()
  # User name
  cfg['user'] = 'qlan3'
  # Project directory
  cfg['project_dir'] = '/home/qlan3/projects/def-afyshe-ab/qlan3/Explorer'
  # Sbatch script path
  cfg['script_path'] = './sbatch.sh'
  # Job indexes list
  cfg['job_list'] = list(range(1,450+1))
  # Check time interval in minutes
  cfg['check_time_interval'] = 10
  # cluster_name: cluster_capacity
  # cfg['clusters'] = {'Mp2':7000, 'Cedar':7000, 'Graham': 1000, 'Beluga':1000}
  cfg['clusters'] = {'Cedar': 7000}

  make_dir('output/minatar_2')
  submitter = Submitter(cfg)
  submitter.submit()

if __name__=='__main__':
  main()