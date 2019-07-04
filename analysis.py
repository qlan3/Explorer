import os
from utils.plot import *

show = False
merged = True
topKRun = 0
imgType = 'png'
test_result_label = 'Average Return'
train_result_label = 'Rolling Return'

def maxmin():
  # game = 'Pixelcopter'
  # game = 'LunarLander'
  # game = 'Catcher'
  for agent in ['MaxminDQN', 'DQN', 'DDQN']:
    for game in ['Pixelcopter', 'Catcher', 'LunarLander']:
      merge_game_agent_allIndex(game, agent, mode='Test')
      merge_game_agent_allIndex(game, agent, mode='Train')
      sort_merged_test_result(game, agent)
      show_results(game, agent, 'Test', test_result_label, show, topKRun, imgType)
      show_results(game, agent, 'Train', train_result_label, show, topKRun, imgType)

if __name__ == "__main__":
  maxmin()
  # unfinished_index('LunarLander', 'MaxminDQN', runs=20)
  '''
  game = 'Pixelcopter'
  agentIndexList = [['DQN', 5], ['DDQN', 5], ['MaxminDQN', 32]]
  plot_game_agentIndexList(game, agentIndexList, game, train_result_label, show, merged, topKRun, imgType)
  
  game = 'Catcher'
  agentIndexList = [['DQN', 4], ['DDQN', 4], ['MaxminDQN', 25]]
  plot_game_agentIndexList(game, agentIndexList, game, train_result_label, show, merged, topKRun, imgType)
  
  game = 'LunarLander'
  agentIndexList = [['DQN', 3], ['DDQN', 3], ['MaxminDQN', 14]]
  plot_game_agentIndexList(game, agentIndexList, game, train_result_label, show, merged, topKRun, imgType)
  '''