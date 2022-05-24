import os
import sys
from utils.submitter import Submitter


def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)


def main(argv):
  sbatch_cfg = {
    # Account name
    # 'account': 'def-ashique',
    'account': 'rrg-ashique',
    # Job name
    # 'job-name': 'minatar_dqn',
    'job-name': 'minatar_dqn_sm',
    # 'job-name': 'minatar_medqn_real',
    # 'job-name': 'minatar_medqn_uniform',
    # Job time
    'time': '0-05:00:00',
    # GPU/CPU type
    'cpus-per-task': 1,
    # Memory
    # 'mem-per-cpu': '2500M',
    'mem-per-cpu': '1500M',
    # Email address
    'mail-user': 'qlan3@ualberta.ca'
  }

  # sbatch configs backup for different games
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'catcher', '0-10:00:00', '2000M'
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'copter', '0-05:00:00', '2000M'
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'lunar', '0-07:00:00', '2000M'
  # sbatch_cfg['job-name'], sbatch_cfg['time'], sbatch_cfg['mem-per-cpu'] = 'minatar', '0-05:00:00', '2500M'


  l_dqn = [11,15,19,7,13,17,9,12,16,6,10,2,18]
  l_dqn.sort()
  ll_dqn = []
  for r in range(1,10):
    for x in l_dqn:
      ll_dqn.append(x+20*r)

  l_dqn_sm = [19,11,15,13,17,9,18,14,2,20,12]
  l_dqn_sm.sort()
  ll_dqn_sm = []
  for r in range(1,10):
    for x in l_dqn_sm:
      ll_dqn_sm.append(x+20*r)

  l_uniform = [827,267,927,155,583,691,751,351,147,747,587,277,269,273,497,357,669,433,501,509,821,205,517,577,254,270,826,746,490,510,730,430,830,734,732,736,652,888,656,892,512,496,572,592,508,352]
  l_uniform.sort()
  ll_uniform = []
  for r in range(1,10):
    for x in l_uniform:
      ll_uniform.append(x+960*r)
  
  l_real = [195,643,27,179,423,115,403,187,191,267,419,43,351,199,31,203,357,121,41,201,125,285,129,133,213,749,429,433,517,493,501,505,489,57,432,888,884,564,648,664,644,188,416,652,276,352,340,108,256,426,106,402,566,110,510,406,410,254,414,206,574]
  l_real.sort()
  ll_real = []
  for r in range(1,10):
    for x in l_real:
      ll_real.append(x+960*r)

  general_cfg = {
    # User name
    'user': 'qlan3',
    # Sbatch script path
    'script-path': './sbatch.sh',
    # Check time interval in minutes
    'check-time-interval': 5,
    # Clusters info: {name: capacity}
    'clusters': {'Narval': 1000},
    # Job indexes list
    # 'job-list': list(range(1, 20+1))
    # 'job-list': list(range(1, 960+1))
    # 'job-list': ll_uniform
    # 'job-list': ll_real
    # 'job-list': ll_dqn
    'job-list': ll_dqn_sm
    # 'job-list': []
  }

  make_dir(f"output/{sbatch_cfg['job-name']}")
  submitter = Submitter(general_cfg, sbatch_cfg)
  submitter.submit()

if __name__=='__main__':
  main(sys.argv)