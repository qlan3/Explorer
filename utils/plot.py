import pandas as pd
import seaborn as sns; sns.set(style='darkgrid')
import matplotlib.pyplot as plt

def plot_agent_results(log_name, image_name, title, y_label):
  df = pd.read_csv(log_name)
  ax = sns.lineplot(x='Episode', y=y_label, hue='Agent', data=df)
  ax.set_title(title)
  ax.get_figure().savefig(image_name)
  plt.show()

def plot_agents_results():
  # TODO
  return


def tb_plot_agent_results(tb_log_dir):
  # TODO
  return