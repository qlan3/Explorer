import os
import time

class Submitter(object):
  def __init__(self, cfg):
    self.user = cfg['user']
    self.project_dir = cfg['project_dir']
    self.script_path = cfg['script_path']
    self.job_list = [str(x) for x in cfg['job_list']]
    self.check_time_interval = cfg['check_time_interval'] * 60
    self.clusters = cfg['clusters']
  
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

    bash_script = f'cd {self.project_dir}; sbatch --array={job_indexes} {self.script_path}'
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
        print(f'Number of current running jobs in {cluster_name}:', num_current_jobs)
        # If cluster capacity is not reached and job list is not empty, submit new jobs
        if (num_current_jobs < cluster_capacity) and (len(self.job_list) > 0):
          num_jobs = min(cluster_capacity - num_current_jobs, len(self.job_list))
          self.submit_jobs(num_jobs, cluster_name)
          if len(self.job_list) == 0:
            print("Finish submitting all jobs!")
            exit(1)
      time.sleep(self.check_time_interval)