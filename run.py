from utils.submitter import Submitter

def main():
  cfg = dict()
  # User name
  cfg['user'] = 'qlan3'
  # Project directory
  cfg['project_dir'] = '/home/qlan3/projects/def-afyshe-ab/qlan3/Explorer'
  # Sbatch script path
  cfg['script_path'] = './sbatch.sh'
  cfg['job_list'] = list(range(1,28+1))
  # Check time interval in minutes
  cfg['check_time_interval'] = 5
  # cluster_name: cluster_capacity
  cfg['clusters'] = {'Mp2': 1000}

  submitter = Submitter(cfg)
  submitter.submit()

if __name__=='__main__':
  main()