This branch is the code for the paper:

**Qingfeng Lan**, Samuele Tosatto, Homayoon Farrahi, A. Rupam Mahmood. **Model-free Policy Learning with Reward Gradients.** AISTATS, 2022. **(Poster)** [[paper]](https://arxiv.org/abs/2103.05147)

Please use this bibtex to cite this paper

```
@inproceedings{lan2022model,
  title={Model-free Policy Learning with Reward Gradients},
  author={Lan, Qingfeng and Tosatto, Samuele and Farrahi, Homayoon and Mahmood, A. Rupam},
  booktitle={The 25th International Conference on Artificial Intelligence and Statistics},
  year={2022}
}
```


## Requirements

- Python (>=3.6)
- [PyTorch](https://pytorch.org/)
- [Gym && Gym Games](https://github.com/qlan3/gym-games): You may only install part of Gym (`classic_control, box2d`) by command `pip install 'gym[classic_control, box2d]'`.
- Optional: 
  - [Gym Atari](https://github.com/openai/gym/blob/master/docs/environments.md#atari)
  - [Gym Mujoco](https://github.com/openai/gym/blob/master/docs/environments.md#mujoco)
  - [PyBullet](https://pybullet.org/): `pip install pybullet`
  - [DeepMind Control Suite](https://github.com/denisyarats/dmc2gym): `pip install git+git://github.com/denisyarats/dmc2gym.git`
- Others: Please check `requirements.txt`.


## Experiments

### Train && Test

All hyperparameters including parameters for grid search are stored in a configuration file in directory `configs`. To run an experiment, a configuration index is first used to generate a configuration dict corresponding to this specific configuration index. Then we run an experiment defined by this configuration dict. All results including log files and the model file are saved in directory `logs`. Please refer to the code for details.

For example, run the experiment with configuration file `rpg.json` and configuration index `1`:

```python main.py --config_file ./configs/rpg.json --config_idx 1```

The models are tested for one episode after every `test_per_epochs` training epochs which can be set in the configuration file.


### Grid Search (Optional)

First, we calculate the number of total combinations in a configuration file (e.g. `rpg.json`):

`python utils/sweeper.py`

The output will be:

`Number of total combinations in rpg.json: 12`

Then we run through all configuration indexes from `1` to `12`. The simplest way is a bash script:

``` bash
for index in {1..12}
do
  python main.py --config_file ./configs/rpg.json --config_idx $index
done
```

[Parallel](https://www.gnu.org/software/parallel/) is usually a better choice for scheduling a large number of jobs:

``` bash
parallel --eta --ungroup python main.py --config_file ./configs/rpg.json --config_idx {1} ::: $(seq 1 12)
```

Any configuration index that has the same remainder (divided by the number of total combinations) should has the same configuration dict. So for multiple runs, we just need to add the number of total combinations to the configuration index. For example, 5 runs for configuration index `1`:

```
for index in 1 13 25 37 49
do
  python main.py --config_file ./configs/rpg.json --config_idx $index
done
```

Or a simpler way:
```
parallel --eta --ungroup python main.py --config_file ./configs/rpg.json --config_idx {1} ::: $(seq 1 12 60)
```

### Analysis (Optional)

To analysis the experimental results, just run:

`python analysis_rpg.py`

Inside `analysis_rpg.py`, `unfinished_index` will print out the configuration indexes of unfinished jobs based on the existence of the result file. `memory_info` will print out the memory usage information and generate a histogram to show the distribution of memory usages in directory `logs/rpg/0`. Similarly, `time_info` will print out the time information and generate a histogram to show the distribution of time in directory `logs/rpg/0`. Finally, `analyze` will generate `csv` files that store training and test results. More functions are available in `utils/plotter.py`.

To reprodce figures in the paper after all jobs are done, simply run:

`python plot.py`

Enjoy!