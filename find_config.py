import os
from utils.sweeper import Sweeper


def find_cfg_idx():
  agent_config = 'MERL_mc_medqn.json'
  config_file = os.path.join('./configs/', agent_config)
  sweeper = Sweeper(config_file)
  for i in range(1, 1+sweeper.config_dicts['num_combinations']):
    cfg = sweeper.generate_config_for_idx(i)
    if cfg['agent']['consod_start'] == cfg['agent']['consod_end']:
      print(i, end=',')
  print()


def get_cfg_idx_for_runs():
  l = [23,146,150,147,255,207,133,130,114,55,235,210,138,82,140,209,228,69,71,353,317]
  l.sort()
  print('len(l)=', len(l))
  ll = []
  for r in range(1,20):
    for x in l:
      ll.append(x+360*r)
  print('len(ll)=', len(ll))
  for x in ll:
    print(x, end=',')


if __name__ == "__main__":
  find_cfg_idx()
  # get_cfg_idx_for_runs()