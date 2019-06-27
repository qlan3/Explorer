import os
from utils.plot import *

def main():
  game = 'Pixelcopter'
  title = game
  agent = 'DQN'
  show = True
  topKRun = 3
  test_result_label = 'Average Return'
  train_result_label = 'Rolling Return'
  '''
  merge_game_agent_allIndex(game, agent, mode='Test')
  merge_game_agent_allIndex(game, agent, mode='Train')
  sort_merged_test_result(game, agent)
  show_results(game, agent, 'Test', test_result_label, show, topKRun)
  show_results(game, agent, 'Train', train_result_label, show, topKRun)
  '''
  config_idx = 1
  merged = False
  indexList = [1,2]
  agentIndexList = [[agent,1],[agent,2]]
  # plot_game_agentIndexList(game, agentIndexList, title, train_result_label, show, merged, topKRun)
  plot_game_agent_index(game, agent, config_idx, title, train_result_label, show, merged, 0)

if __name__ == "__main__":
  main()