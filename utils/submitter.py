import os
import time


class Submitter(object):
  def __init__(self, general_cfg, sbatch_cfg):
    general_cfg.setdefault('max_parallel_jobs', 1)
    self.user = general_cfg['user']
    self.script_path = general_cfg['script-path']
    self.job_list = [str(x) for x in general_cfg['job-list']]
    self.check_time_interval = general_cfg['check-time-interval'] * 60
    self.cluster_name = general_cfg['cluster_name']
    self.cluster_capacity = general_cfg['cluster_capacity']
    self.max_parallel_jobs = general_cfg['max_parallel_jobs']
    self.general_cfg = general_cfg
    self.sbatch_cfg = sbatch_cfg
  
  def submit_multiple_jobs(self, num_jobs):
    # Write job indexes into file
    job_indexes = '\n'.join(self.job_list[:num_jobs])
    with open(f'job_idx_{self.sbatch_cfg["job-name"]}_{self.job_list[0]}.txt', 'w') as f:
      f.write(job_indexes)
    # Sleep for a few seconds to allow file saving
    time.sleep(2)
    slurm_input = ' '.join([f'--{k}={v}' for k, v in self.sbatch_cfg.items()])
    bash_script = f'sbatch --array={self.job_list[0]} {slurm_input} {self.script_path}'
    myCmd = os.popen(bash_script).read()
    print(myCmd)
    print(f'Submit jobs from {self.job_list[0]} to {self.job_list[num_jobs-1]}')
    # Pop jobs in job list
    self.job_list = self.job_list[num_jobs:]
  
  def multiple_submit(self):
    while True:
      bash_script = f'squeue -u {self.user} -r'
      myCmd = os.popen(bash_script).read()
      lines = myCmd.split('\n')
      num_current_jobs = 0
      for line in lines:
        if self.user in line:
          num_current_jobs += 1
      print(f'Number of current jobs in {self.cluster_name}:', num_current_jobs)
      # If cluster capacity is not reached and job list is not empty, submit new jobs
      if (num_current_jobs < self.cluster_capacity) and (len(self.job_list) > 0):
        num_jobs = min(self.cluster_capacity - num_current_jobs, len(self.job_list))
        while num_jobs > 0:
          if num_jobs >= self.max_parallel_jobs:
            self.submit_multiple_jobs(self.max_parallel_jobs)
            num_jobs -= self.max_parallel_jobs
          else:
            self.submit_multiple_jobs(num_jobs)
            num_jobs = 0
        if len(self.job_list) == 0:
          print("Finish submitting all jobs!")
          exit(1)
      time.sleep(self.check_time_interval)
      

  def submit_single_jobs(self, num_jobs):
    # Get job indexes and submit jobs
    flag = True
    for i in range(num_jobs):
      if int(self.job_list[i]) != int(self.job_list[0]) + i:
        flag = False
        break
    if flag == True:
      job_indexes = f'{self.job_list[0]}-{self.job_list[num_jobs-1]}'
    else:
      job_indexes = ','.join(self.job_list[:num_jobs])
    slurm_input = ' '.join([f'--{k}={v}' for k, v in self.sbatch_cfg.items()])
    bash_script = f'sbatch --array={job_indexes} {slurm_input} {self.script_path}'
    myCmd = os.popen(bash_script).read()
    print(myCmd)
    print(f'Submit jobs from {self.job_list[0]} to {self.job_list[num_jobs-1]}')
    # Pop jobs in job list
    self.job_list = self.job_list[num_jobs:]
  
  def single_submit(self):
    while True:
      bash_script = f'squeue -u {self.user} -r'
      myCmd = os.popen(bash_script).read()
      lines = myCmd.split('\n')
      num_current_jobs = 0
      for line in lines:
        if self.user in line:
          num_current_jobs += 1
      print(f'Number of current jobs in {self.cluster_name}:', num_current_jobs)
      # If cluster capacity is not reached and job list is not empty, submit new jobs
      if (num_current_jobs < self.cluster_capacity) and (len(self.job_list) > 0):
        num_jobs = min(self.cluster_capacity - num_current_jobs, len(self.job_list))
        if num_jobs > 0:
          self.submit_single_jobs(num_jobs)
        if len(self.job_list) == 0:
          print("Finish submitting all jobs!")
          exit(1)
      time.sleep(self.check_time_interval)