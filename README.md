# Explorer

Explorer is a reinforcement learning frame based on Pytorch for **exploring** new ideas.


# Implemented algorithms

- Vanilla Deep Q-learning (VanillaDQN): No target network.
- [Deep Q-Learning (DQN)](https://users.cs.duke.edu/~pdinesh/sources/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double Deep Q-learning (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Maxmin Deep Q-learning (MaxminDQN)](https://openreview.net/pdf?id=Bkg0u3Etwr)
- [Averaged Deep Q-learning (AveragedDQN)](https://arxiv.org/pdf/1611.01929.pdf)
- [Ensemble Deep Q-learning (EnsembleDQN)](https://arxiv.org/pdf/1611.01929.pdf)
- [REINFORCE](http://incompleteideas.net/book/RLbook2020.pdf)
- [REINFORCE with baseline](http://incompleteideas.net/book/RLbook2020.pdf)
- [Actor-Critic](http://incompleteideas.net/book/RLbook2020.pdf)
- [Syncrhonous Advantage Actor-Critic (A2C)](https://arxiv.org/pdf/1602.01783.pdf)
- [Proximal Policy Optimisation (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/pdf/1802.09477.pdf)

## To do list
- Add Mujuco
- Add PyBullet
- Add more pygames to gym-games

## Requirements

- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- [Gym && Gym Games](https://github.com/qlan3/gym-games): You may only install part of Gym (`classic_control, box2d, atari`) by command `pip install 'gym[classic_control, box2d, atari]'`.
- Others: Please check `requirements.txt`.


## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files and the model file are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `catcher.json` and configuration index `1`:

```python main.py --config_file ./configs/catcher.json --config_idx 1```

The models are tested for one episode after every `test_per_episodes` training episodes which can be set in the configuration file.


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `catcher.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in catcher.json: 90`

Then we run through all configuration indexes from `1` to `90`. The simplest way is a bash script:

``` bash
for index in {1..90}
do
  python main.py --config_file ./configs/catcher.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice for scheduling a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/catcher.json --config_idx {1} ::: $(seq 1 90)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should has the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```
for index in 1 91 181 271 361
do
  python main.py --config_file ./configs/catcher.json --config_idx $index
done
```

Or a simpler way:
```
parallel --eta --ungroup python main.py --config_file ./configs/catcher.json --config_idx {1} ::: $(seq 1 90 450)
```

To reproduce the results in the [Maxmin Q-learning paper](https://openreview.net/forum?id=Bkg0u3Etwr&noteId=Bkg0u3Etwr), please run experiments with the configuration files provided in directory `configs` (except for `atari_ram.json` which is experimental).

### Analysis (Optional)

To analysis the experimental results, just run:

`python analysis.py`

Inside `analysis.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in directory `logs/catcher/0`. Similarly, `time_info` will print out the time information and generate a histogram to show the distribution of time in directory `logs/catcher/0`. Finally, `analyze` will generate `csv` files that store training and test results. More functions are available in `utils/plotter.py`.

Enjoy!


### Cite

Please use this bibtex to cite this repo

```
@misc{Explorer,
  author = {Lan, Qingfeng},
  title = {A PyTorch Reinforcement Learning Frame for Exploring New Ideas},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/qlan3/Explorer}}
}
```

# Acknowledgements

- [DeepRL](https://github.com/ShangtongZhang/DeepRL)
- [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
- [Classic Control](https://github.com/muhammadzaheer/classic-control)
- [Spinning Up in Deep RL](https://github.com/openai/spinningup)