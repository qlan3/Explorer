import os
import sys
import argparse

from utils.submitter import Submitter
#from utils.helper import make_dir

def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)

def main(argv):
  # python run.py --job_name minatar_2 --script_path  ./sbatch_minatar_2.sh
  # python run.py --job_name catcher --script_path  ./sbatch_catcher.sh
  # python run.py --job_name lunar --script_path  ./sbatch_lunar.sh
  # python run.py --job_name copter --script_path  ./sbatch_copter.sh
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--user', type=str, default='qlan3', help='user name')
  parser.add_argument('--project_dir', type=str, default='/home/qlan3/projects/def-afyshe-ab/qlan3/Explorer', help='project directory')
  parser.add_argument('--job_name', type=str, default='minatar_2', help='job name')
  parser.add_argument('--script_path', type=str, default='./sbatch.sh', help='sbatch script path')
  parser.add_argument('--check_time_interval', type=int, default=10, help='check time interval in minutes')
  args = parser.parse_args()
  
  cfg = dict()
  # User name
  cfg['user'] = args.user
  # Project directory
  cfg['project_dir'] = args.project_dir
  # Sbatch script path
  cfg['script_path'] = args.script_path
  # Job indexes list
  cfg['job_list'] = list(range(1,450+1))
  # Check time interval in minutes
  cfg['check_time_interval'] = args.check_time_interval
  # cluster_name: cluster_capacity
  # cfg['clusters'] = {'Mp2':7000, 'Cedar':7000, 'Graham': 1000, 'Beluga':1000}
  cfg['clusters'] = {'Cedar': 7000}

  make_dir(f'output/{args.job_name}')
  submitter = Submitter(cfg)
  submitter.submit()

if __name__=='__main__':
  main(sys.argv)