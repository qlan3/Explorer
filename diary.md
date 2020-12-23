# Reward Policy Gradient

- Github repo: https://github.com/qlan3/Explorer_in_progress
- Log: https://drive.google.com/drive/folders/1cyR19_dlgWkw7d9tNjAdQBgR21lOqrMt
- Experiment record template:

## 2020-12-xx

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   onrpg    |  onrpg.json |   1  |        |   RPG  | e20a73b |

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
    - onrpg2: A2C style, on-policy
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