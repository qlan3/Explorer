import os
from utils.plot import *

def maxmin():
  show = False
  topKRun = 0
  test_result_label = 'Average Return'
  train_result_label = 'Rolling Return'
  
  game = 'Pixelcopter'
  agent = 'DQN'
  merge_game_agent_allIndex(game, agent, mode='Test')
  merge_game_agent_allIndex(game, agent, mode='Train')
  sort_merged_test_result(game, agent)
  show_results(game, agent, 'Test', test_result_label, show, topKRun)
  show_results(game, agent, 'Train', train_result_label, show, topKRun)

if __name__ == "__main__":
  maxmin()