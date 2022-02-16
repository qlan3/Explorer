import os
from utils.sweeper import Sweeper

agent_config = 'catcher.json'
config_file = os.path.join('./configs/', agent_config)
sweeper = Sweeper(config_file)

# Find cfg index with certain constraint
for i in range(1, 1+sweeper.config_dicts['num_combinations']):
  cfg = sweeper.generate_config_for_idx(i)
  if cfg['agent']['name'] == 'MaxminDQN':
    print(i, end=',')
print()