# Explorer

Explorer is a PyTorch reinforcement learning framework for **exploring** new ideas.


## Implemented algorithms

- Vanilla Deep Q-learning (VanillaDQN): No target network.
- [Deep Q-Learning (DQN)](https://users.cs.duke.edu/~pdinesh/sources/MnihEtAlHassibis15NatureControlDeepRL.pdf)
- [Double Deep Q-learning (DDQN)](https://arxiv.org/pdf/1509.06461.pdf)
- [Maxmin Deep Q-learning (MaxminDQN)](https://arxiv.org/pdf/2002.06487.pdf)
- [Averaged Deep Q-learning (AveragedDQN)](https://arxiv.org/pdf/1611.01929.pdf)
- [Ensemble Deep Q-learning (EnsembleDQN)](https://arxiv.org/pdf/1611.01929.pdf)
- [Bootstrapped Deep Q-learning (BootstrappedDQN)](https://arxiv.org/pdf/1602.04621.pdf)
- [NoisyNet Deep Q-learning (NoisyNetDQN)](https://arxiv.org/pdf/1706.10295.pdf)
- [REINFORCE](http://incompleteideas.net/book/RLbook2020.pdf)
- [Actor-Critic](http://incompleteideas.net/book/RLbook2020.pdf)
- [Proximal Policy Optimisation (PPO)](https://arxiv.org/pdf/1707.06347.pdf)
- [Soft Actor-Critic (SAC)](https://arxiv.org/pdf/1812.05905.pdf)
- [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/pdf/1509.02971.pdf)
- [Twin Delayed Deep Deterministic Policy Gradients (TD3)](https://arxiv.org/pdf/1802.09477.pdf)
- [Reward Policy Gradient (RPG)](https://arxiv.org/pdf/2103.05147.pdf)
- [Memory-efficient Deep Q-learning (MeDQN)](https://arxiv.org/pdf/2205.10868.pdf)

## To do list

- SAC with automatically adjusted temperature
- SAC with discrete action spaces

## The dependency tree of agent classes

    Base Agent
      ├── Vanalla DQN
      |     ├── DQN
      |     |    ├── DDQN
      |     |    ├── NoisyNetDQN
      |     |    ├── BootstrappedDQN
      |     |    └── MeDQN_Uniform, MeDQN_Real
      |     ├── Maxmin DQN ── Ensemble DQN
      |     └── Averaged DQN
      └── REINFORCE 
            ├── Actor-Critic
            |     └── PPO ── RPG
            └── SAC ── DDPG ── TD3


## Requirements

- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- [Gym && Gym Games](https://github.com/qlan3/gym-games): You may only install part of Gym (`classic_control, box2d`) by command `pip install 'gym[classic_control, box2d]'`.
- Optional: 
  - [Gym Atari](https://www.gymlibrary.ml/environments/atari/): `pip install gym[atari,accept-rom-license]`
  - [Gym Mujoco](https://www.gymlibrary.ml/environments/mujoco/):
    - Download MuJoCo version 1.50 from [MuJoCo website](https://www.roboti.us/download.html).
    - Unzip the downloaded `mjpro150` directory into `~/.mujoco/mjpro150`, and place the activation key (the `mjkey.txt` file downloaded from [here](https://www.roboti.us/license.html)) at `~/.mujoco/mjkey.txt`.
    - Install [mujoco-py](https://github.com/openai/mujoco-py): `pip install 'mujoco-py<1.50.2,>=1.50.1'`
    - Install gym[mujoco]: `pip install gym[mujoco]`
  - [PyBullet](https://pybullet.org/): `pip install pybullet`
  - [DeepMind Control Suite](https://github.com/denisyarats/dmc2gym): `pip install git+git://github.com/denisyarats/dmc2gym.git`
- Others: Please check `requirements.txt`.


## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `RPG.json` and configuration index `1`:

```python main.py --config_file ./configs/RPG.json --config_idx 1```

The models are tested for one episode after every `test_per_episodes` training episodes which can be set in the configuration file.


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `RPG.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in RPG.json: 12`

Then we run through all configuration indexes from `1` to `12`. The simplest way is using a bash script:

``` bash
for index in {1..12}
do
  python main.py --config_file ./configs/RPG.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice to schedule a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/RPG.json --config_idx {1} ::: $(seq 1 12)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should have the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```
for index in 1 13 25 37 49
do
  python main.py --config_file ./configs/RPG.json --config_idx $index
done
```

Or a simpler way:
```
parallel --eta --ungroup python main.py --config_file ./configs/RPG.json --config_idx {1} ::: $(seq 1 12 60)
```


### Analysis (Optional)

To analyze the experimental results, just run:

`python analysis.py`

Inside `analysis.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in directory `logs/RPG/0`. Similarly, `time_info` will print out the time information and generate a histogram to show the distribution of time in directory `logs/RPG/0`. Finally, `analyze` will generate `csv` files that store training and test results. Please check `analysis.py` for more details. More functions are available in `utils/plotter.py`.

Enjoy!


## Code of My Papers

- **Qingfeng Lan**, Yangchen Pan, Alona Fyshe, Martha White. **Maxmin Q-learning: Controlling the Estimation Bias of Q-learning.** ICLR, 2020. **(Poster)** [[paper]](https://openreview.net/pdf?id=Bkg0u3Etwr) [[code]](https://github.com/qlan3/Explorer/releases/tag/maxmin1.0)

- **Qingfeng Lan**, Samuele Tosatto, Homayoon Farrahi, A. Rupam Mahmood. **Model-free Policy Learning with Reward Gradients.** AISTATS, 2022. **(Poster)** [[paper]](https://arxiv.org/pdf/2103.05147.pdf) [[code]](https://github.com/qlan3/Explorer/tree/RPG)

- **Qingfeng Lan**, Yangchen Pan, Jun Luo, A. Rupam Mahmood. **Memory-efficient Reinforcement Learning with Knowledge Consolidation.** Arxiv [[paper]](https://arxiv.org/pdf/2205.10868.pdf) [[code]](https://github.com/qlan3/Explorer/)

## Cite

If you find this repo useful to your research, please cite my paper if related. Otherwise, please cite this repo:

~~~bibtex
@misc{Explorer,
  author = {Lan, Qingfeng},
  title = {A PyTorch Reinforcement Learning Framework for Exploring New Ideas},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/qlan3/Explorer}}
}
~~~

# Acknowledgements

- [DeepRL](https://github.com/ShangtongZhang/DeepRL)
- [Deep Reinforcement Learning Algorithms with PyTorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
- [Classic Control](https://github.com/muhammadzaheer/classic-control)
- [Spinning Up in Deep RL](https://github.com/openai/spinningup)
- [Randomized Value functions](https://github.com/facebookresearch/RandomizedValueFunctions)
- [Rainbow](https://github.com/Kaixhin/Rainbow)