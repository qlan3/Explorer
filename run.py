import os
import sys
import argparse
from math import ceil
from utils.submitter import Submitter


def make_dir(dir):
  if not os.path.exists(dir): 
    os.makedirs(dir, exist_ok=True)


def main(argv):
  # python run_narval.py --job_type S
  # python run_narval.py --job_type M
  parser = argparse.ArgumentParser(description="Submit jobs")
  parser.add_argument('--job_type', type=str, default='S', help='Run single (S) or multiple (M) jobs in one experiment: S, M')
  args = parser.parse_args()

  sbatch_cfg = {
    # Account name
    # 'account': 'def-ashique',
    'account': 'rrg-ashique',
    # Job name
    'job-name': 'MERL_mc_dqn',
    # Job time
    'time': '0-01:00:00',
    # Email notification
    'mail-user': 'qlan3@ualberta.ca'
  }
  general_cfg = {
    # User name
    'user': 'qlan3',
    # Check time interval in minutes
    'check-time-interval': 5,
    # Clusters info: name & capacity
    'cluster_name': 'Narval',
    'cluster_capacity': 996,
    # Job indexes list
    'job-list': list(range(1, 10+1))
  }
  make_dir(f"output/{sbatch_cfg['job-name']}")

  if args.job_type == 'M':
    # Max number of parallel jobs in one experiment
    max_parallel_jobs = 4
    mem_per_job = 16 # in GB
    cpu_per_job = 2 # Increase cpus_per_job to 5/10 can further increase training speed.
    mem_per_cpu = int(ceil(mem_per_job/cpu_per_job))
    # Write to procfile for Parallel
    with open('procfile', 'w') as f:
      f.write(str(max_parallel_jobs))
    sbatch_cfg['gres'] = 'gpu:1' # GPU type
    sbatch_cfg['cpus-per-task'] = cpu_per_job*max_parallel_jobs
    sbatch_cfg['mem-per-cpu'] = f'{mem_per_cpu}G' # Memory
    # Sbatch script path
    general_cfg['script-path'] = './sbatch_m.sh'
    # Max number of jobs for Parallel
    general_cfg['max_parallel_jobs'] = max_parallel_jobs
    submitter = Submitter(general_cfg, sbatch_cfg)
    submitter.multiple_submit()
  elif args.job_type == 'S':
    mem_per_cpu = 1500
    sbatch_cfg['cpus-per-task'] = 1
    sbatch_cfg['mem-per-cpu'] = f'{mem_per_cpu}M' # Memory
    # Sbatch script path
    general_cfg['script-path'] = './sbatch_s.sh'
    submitter = Submitter(general_cfg, sbatch_cfg)
    submitter.single_submit()


if __name__=='__main__':
  main(sys.argv)