import os
import time


class Submitter(object):
  def __init__(self, general_cfg, sbatch_cfg):
    self.user = general_cfg['user']
    self.script_path = general_cfg['script-path']
    self.job_list = [str(x) for x in general_cfg['job-list']]
    self.check_time_interval = general_cfg['check-time-interval'] * 60
    self.clusters = general_cfg['clusters']
    self.general_cfg = general_cfg
    self.sbatch_cfg = sbatch_cfg
  
  def submit_jobs(self, num_jobs, cluster_name):
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
  
  def submit(self):
    while True:
      for cluster_name, cluster_capacity in self.clusters.items():
        bash_script = f'squeue -u {self.user} -r'
        myCmd = os.popen(bash_script).read()
        lines = myCmd.split('\n')
        num_current_jobs = 0
        for line in lines:
          if self.user in line:
            num_current_jobs += 1
        print(f'Number of current jobs in {cluster_name}:', num_current_jobs)
        # If cluster capacity is not reached and job list is not empty, submit new jobs
        if (num_current_jobs < cluster_capacity) and (len(self.job_list) > 0):
          num_jobs = min(cluster_capacity - num_current_jobs, len(self.job_list))
          self.submit_jobs(num_jobs, cluster_name)
          if len(self.job_list) == 0:
            print("Finish submitting all jobs!")
            exit(1)
      time.sleep(self.check-time-interval)