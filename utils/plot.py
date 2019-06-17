import pandas as pd
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt

def show_test_result(log_name, y_label):
  y_label = 'Average ' + y_label
  df = pd.read_csv(log_name)
  y_mean = df[y_label].mean()
  y_std = df[y_label].std(ddof=0)
  print(f'{y_label}: mean={y_mean}, std={y_std}')
  return y_mean, y_std

def plot_results(log_name, image_name, title, y_label, show=True):
  df = pd.read_csv(log_name)
  ax = sns.lineplot(x='Episode', y=y_label, hue='Agent', data=df)
  ax.set_title(title)
  ax.get_figure().savefig(image_name)
  if show: plt.show()

def plot_game_agent_results(game, agent, log_dir, y_label, show=True):
  # TODO
  return

def plot_game_agents_results():
  # TODO
  return

def tb_plot_agent_results(tb_log_dir):
  # TODO
  return