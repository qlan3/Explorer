# Reward Policy Gradient

- Github repo: https://github.com/qlan3/Explorer_in_progress
- Log: https://drive.google.com/drive/folders/1cyR19_dlgWkw7d9tNjAdQBgR21lOqrMt
- Experiment record template:

## 2020-12-xx

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   onrpg    |  onrpg.json |   1  | onrpg  |   RPG  |  |

  - Goal:
  - Analysis:
  - Next:

## 2020-12-15

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   offrpg   | mujoco_offrpg.json |  1   |  mujoco_offrpg  |  RPG   | e20a73b |
|   onrpg1   | mujoco_onrpg1.json |  1   |  mujoco_onrpg1  |  RPG   | e20a73b |
|   onrpg2   | mujoco_onrpg2.json |  1   |  mujoco_onrpg2  |  RPG   | e20a73b |
|   onrpg3   | mujoco_onrpg3.json |  1   |  mujoco_onrpg3  |  RPG   | e20a73b |

  - Goal: Test on-policy and off-policy RPG on Mujoco
    - offrpg: SAC style, off-policy
    - onrpg1: PPO style, on-policy
    - onrpg2: Actor-Critic style, on-policy
    - onrpg3: REINFORCE style, on-policy
  - Analysis:
    - Still, there is `nan` bug for offrpg sometimes.
    - RPG is highly unstable.
    - onrpg3 is better than onrpg1 & onrpg2, probably due to longer episode length (batch size). onrpg 1 & onrpg2 have 64 as the batch size while the average episode length is larger than 64.
  - Next: try larger batch sizes; use try gradient (i.e. include $\gamma^t$ term)


## 2020-12-16

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   offrpg   | mujoco_offrpg.json |  1   |  mujoco_offrpg  |  RPG   | 9f27c34 |
|   onrpg1   | mujoco_onrpg1.json |  1   |  mujoco_onrpg1  |  RPG   | 9f27c34 |
|   onrpg2   | mujoco_onrpg2.json |  1   |  mujoco_onrpg2  |  RPG   | 9f27c34 |

  - Goal: Test on-policy and off-policy RPG with a larger batch size
  - Analysis: A larger batch size didn't seem to help a lot.
  - Next: use true gradient (i.e. include $\gamma^t$ term); delay actor update


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   onrpg1   | mujoco_onrpg1.json |  1   |  mujoco_onrpg1  |  RPG   | cec1781 |
|   onrpg2   | mujoco_onrpg2.json |  1   |  mujoco_onrpg2  |  RPG   | cec1781 |
|   onrpg3   | mujoco_onrpg3.json |  1   |  mujoco_onrpg3  |  RPG   | cec1781 |

  - Goal: Test on-policy RPG with the true gradient and different actor update frequencices
  - Analysis: doesn't seem to help
  - Next: try different actor update frequency, alone.


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   onrpg1   | mujoco_onrpg1.json |  1   |  mujoco_onrpg1  |  RPG   | 0e0ca92 |
|   onrpg2   | mujoco_onrpg2.json |  1   |  mujoco_onrpg2  |  RPG   | 0e0ca92 |
|   onrpg3   | mujoco_onrpg3.json |  1   |  mujoco_onrpg3  |  RPG   | 0e0ca92 |

  - Goal: Test on-policy RPG with different actor update frequencies
  - Analysis: doesn't seem to help
  - Next: add entropy regularization? debug?


## 2020-12-22

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   onrpg4   | onrpg4.json |   5  | mujoco_onrpg4 |  RPG  | 1241e9e |

  - Goal: Test another variant of onrpg: update critic and reward for mutliple batches but only update actor once per epoch.
  - Analysis: no good
  - Next: need more and detailed analysis


## 2020-12-23

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   onrpg32   | onrpg32.json |  5  | mujoco_onrpg32 |  RPG  | 64e3566 |

  - Goal: Test a variant of onrpg3: use rsample
  - Analysis: no good
  - Next: need more and detailed analysis


## 2020-12-24

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_actorcritic  | test_actorcritic.json |  1  | test_actorcritic |  RPG  | d3fa59c |
|  test_ppo  | test_ppo.json |  1  | test_ppo |  RPG  | d3fa59c |
|  test_onrpg  | test_onrpg.json |  1  | test_onrpg |  RPG  | d3fa59c |

  - Goal: 
    - analysis: plot V, log_prob, actor loss, critic loss of Actor-Critic, PPO, and OnRPG2 during training, on HalfCheetah
    - test the influence of the discount factor
  - Analysis:
    - For discount factor: 0.99 is much better than 1; in fact, when discount factor = 1, Actor-Critic cannot even learn!
    - For Actor-Critic: V (down then up), log_prob (up to -8), actor loss (up then flat), critic loss (down to 100)
    - For PPO: V (down then up), log_prob (up to -5/2), actor loss (flat), critic loss (down to >200)
    - For OnRPG: V (keep going straight down), log_prob(flat -8), actor loss (down), critic loss (flat 2), reward loss (down to 0.04)
    - In general, to get good performance, we need low critic loss, high log_prob (high PDF & low variance), high V
  - Next: use normalized state values (we want to increase the log_prob of an action that has **adavantage**)


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_onrpg    |  test_onrpg.json |   1  |  onrpg   |   RPG  | 339187c |
|   test_onrpg2   |  test_onrpg2.json |  1  |  onrpg2  |   RPG  | 339187c |

  - Goal: test two ways to normalize the state values V: OnRPG (subtract mean of V), OnRPG2 (subtract mean of V, then divided by standard deviation)
  - Analysis: OnRPG2 learns faster and achieves higher performance in some environments.
  - Next: use OnRPG2's way to normalize V, implement PPO style OnRPG


## 2020-12-25

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_onrpg3    |  test_onrpg3.json |   1  |   test_onrpg3   |   RPG  | 339187c |
|   test_onrpg4    |  test_onrpg4.json |   1  |   test_onrpg4   |   RPG  | 339187c |

  - Goal: test PPO style OnRPG (OnRPG3: V divided by std; OnRPG4: V not divided by std)
  - Analysis: OnRPG4 is better than OnRPG3 for 4 games, worse than OnRPG3 for 2 games. Both are worse than PPO.
  - Next: also use ratio clip for the reward part; use larger lr for actor


## 2020-12-26

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_onrpg5    |  test_onrpg5.json |   1  |   test_onrpg5   |   RPG  | 339187c |
|   test_onrpg6    |  test_onrpg6.json |   1  |   test_onrpg6   |   RPG  | 339187c |

  - Goal: test PPO style OnRPG (OnRPG5: V divided by std; OnRPG6: V not divided by std), also use ratio clip for the reward part
  - Analysis: OnRPG6 is better than OnRPG5 for 3 games, worse than OnRPG5 for 3 games. Both are worse than PPO.
  - Next: search over different choices, search actor lr.


## 2020-12-27

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_onrpg    |  test_onrpg.json |   1  | test_onrpg |  RPG  | d5a56c7 |

  - Goal: test all combinations of choices: divide adv by std, clip reward, clip adv; search actor lr.
  - Analysis: inconsistent results
  - Next: run with less optimize epoch; add state normalizer.


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_ppo  | test_ppo.json |  1  | test_ppo |  RPG  | d5a56c7 |

  - Goal: test with less optimize epoch and different gae: [0.95, 0.97]
  - Analysis: inconsistent results
  - Next: add state normalizer.


## 2020-12-28

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_offrpg   |  test_offrpg.json |   1  | test_offrpg  |   RPG  |  |

  - Goal: test DDPG style off-policy RPG, select different actors
  - Analysis: too bad
  - Next: try SAC style


## 2020-12-29

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_offrpg2    |  test_offrpg2.json |   1  | test_offrpg2  |   RPG  |  |

  - Goal: test SAC style off-policy RPG, select different actors
  - Analysis: even worse than DDPG style of RPG.
  - Next: off-policy without importance sampling may not be a good option


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_ppo  | test_ppo.json |  1  | test_ppo |  RPG  |  |

  - Goal: test ppo with a MeanStdNormalizer state normalizer, a different way (use `softplus` rather than `exp`) to set action_std.
  - Analysis: `softplus` is better than `exp`; state normalizer can be dropped.
  - Next: test spinning up version of PPO using `softplus`; drop state normalizer


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_ppo2  | test_ppo2.json |  1  | test_ppo2 |  RPG  |  |

  - Goal: test spinning up version of PPO using `softplus`
  - Analysis: some are better than original PPO, some are worse
  - Next: abandon this version.