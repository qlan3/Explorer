import os
import sys
from utils.submitter import Submitter


def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)


def main(argv):
   
  sbatch_cfg = {
    # Account name
    'account': 'rrg-whitem',
    # Job name
    'job-name': 'catcher',
    # Job time
    'time': '0-10:00:00',
    # GPU/CPU type
    'cpus-per-task': 1,
    # Memory
    'mem-per-cpu': '2000M',
    # Email address
    'mail-user': 'qlan3@ualberta.ca'
  }

  # sbatch configs backup for different games
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'catcher', '0-10:00:00', '2000M'
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'copter', '0-05:00:00', '2000M'
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'lunar', '0-07:00:00', '2000M'
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'minatar', '1-08:00:00', '4000M'
  
  
  general_cfg = {
    # User name
    'user': 'qlan3',
    # Sbatch script path
    'script-path': './sbatch.sh',
    # Check time interval in minutes
    'check-time-interval': 5,
    # Clusters info: {name: capacity}
    'clusters': {'Cedar': 3000},
    # Job indexes list
    'job-list': list(range(1, 30+1))
   }

  make_dir(f"output/{sbatch_cfg['job-name']}")
  submitter = Submitter(general_cfg, sbatch_cfg)
  submitter.submit()

if __name__=='__main__':
  main(sys.argv)