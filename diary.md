# Reward Policy Gradient

- Github repo: https://github.com/qlan3/Explorer_in_progress
- Log: https://drive.google.com/drive/folders/1cyR19_dlgWkw7d9tNjAdQBgR21lOqrMt
- Experiment record template:

## 2021-01-xx

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|   onrpg    |   1  |   RPG  |  |

  - Goal:
  - Analysis:

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
  - Next: try larger batch sizes; use true gradient (i.e. include $\gamma^t$ term)


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
    - analysis: plot V, log_pi, actor loss, critic loss of Actor-Critic, PPO, and OnRPG2 during training, on HalfCheetah
    - test the influence of the discount factor
  - Analysis:
    - For discount factor: 0.99 is much better than 1; in fact, when discount factor = 1, Actor-Critic cannot even learn!
    - For Actor-Critic: V (down then up), log_pi (up to -8), actor loss (up then flat), critic loss (down to 100)
    - For PPO: V (down then up), log_pi (up to -5/2), actor loss (flat), critic loss (down to >200)
    - For OnRPG: V (keep going straight down), log_pi(flat -8), actor loss (down), critic loss (flat 2), reward loss (down to 0.04)
    - In general, to get good performance, we need low critic loss, high log_pi (high PDF & low variance), high V
  - Next: use normalized state values (we want to increase the log_pi of an action that has **adavantage**)


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
|   test_offrpg   |  test_offrpg.json |   1  | test_offrpg  |   RPG  | d102899 |

  - Goal: test DDPG style off-policy RPG, select different actors
  - Analysis: too bad
  - Next: try SAC style


## 2020-12-29

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|   test_offrpg2    |  test_offrpg2.json |   1  | test_offrpg2  |   RPG  | d102899 |

  - Goal: test SAC style off-policy RPG, select different actors
  - Analysis: even worse than DDPG style of RPG.
  - Next: off-policy without importance sampling may not be a good option


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_ppo  | test_ppo.json |  1  | test_ppo |  RPG  | d102899 |

  - Goal: test ppo with a MeanStdNormalizer state normalizer, a different way (use `softplus` rather than `exp`) to set action_std.
  - Analysis: `softplus` is better than `exp`; state normalizer can be dropped.
  - Next: test spinning up version of PPO using `softplus`; drop state normalizer


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_ppo2  | test_ppo2.json |  1  | test_ppo2 |  RPG  | d102899 |

  - Goal: test spinning up version of PPO using `softplus`
  - Analysis: some are better than original PPO, some are worse
  - Next: abandon this version.


## 2020-12-30

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg | test_onrpg.json |  1  | test_onrpg  |   RPG  | 7946077 |

  - Goal: test new OnRPG: search actor lr, divide advantage by std or not, clip objective or not, use a state normalizer or not, select actor type
  - Analysis:
    - actor select: MLPGaussianActor is good enough
    - actor lr: 1e-4 or 3e-4
    - clip objective: seems to be neutral? (not clear)
    - divide advantage by std: not clear
    - state normalizer: doesn't hurt, sometimes is helpful, better to include
    - Hopper is too noise as a test environment, use Walker2d instead
  - Next: try different optimization epochs; use Walker2d; try off-policy RPG


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
|  test_onrpg  | test_onrpg.json |  1  | test_onrpg  |  RPG  | 6799aff |
|  test_onrpg2  | test_onrpg2.json |  1  | test_onrpg2  |  RPG  | 6799aff |
|  test_offrpg | test_offrpg.json |  1 | test_offrpg |  RPG  | 6799aff |

  - Goal: for OnRPG, test different advantages and objectives, use state normalizer or not; for OffRPG, test different advantages, actors and v_next, use state normalizer or not.
  - Analysis:
    - OffRPG:
      - MLPSquashedGaussianActor still has NaN problem, why? Look into this.
      - The learning process is stuck at a low performance. We may need IS and better exploration.
    - OnRPG:
      - use state normalizer is good.
      - still not clear about different advantages and objectives.
  - Next: more detailed analysis.


## 2021-01-04

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg | test_onrpg.json |  1  | test_onrpg  |   RPG  | 54e4a4e |
| test_ppo   | test_ppo.json   |  1  | test_ppo    |   RPG  | 54e4a4e |

  - Goal: plot log_pi, entropy, action_std, KL, IS, state value, advantage (original advantage)
  - Analysis: PPO doesn't use TD error to optimize the critic (i.e. state value function), but use the true return!
    - PPO:
    - OnRPG: KL and IS are larger; actor loss, (original) advantage, and state value function are much unstable; critic loss is much smaller than PPO, i.e. 10 vs 1000. log_pi, entropy, and action_std seem to be good.
  - Next: use the true return to compute critic loss; try different clip_ratio and target_kl


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg | test_onrpg.json |  1  | test_onrpg  |   RPG  | 29e3a0e |
  
  - Goal: use the true return to compute critic loss; try gradient_clip
  - Analysis: slightly better.
  - Next: use lambda return.


## 2021-01-05

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg1 | test_onrpg1.json |  1  | test_onrpg1  |   RPG  | 29e3a0e |
  
  - Goal: test OnRPG1 with lambda return, gradient_clip, two baselines (mean lambda return v.s. V)
  - Analysis: gradient clip, lambda return helps; baseline V is better than the mean lambda return.
  - Next: try baseline V for previous OnRPG


## 2021-01-06

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg2 | test_onrpg2.json |  1  | test_onrpg2  |   RPG  | 29e3a0e |
  
  - Goal: test OnRPG with gradient_clip and baseline V
  - Analysis: baseline V is much better
  - Next: try normalize adv after subtrace the baseline; test with more games and without state normalizer


## 2021-01-07

| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg | test_onrpg.json |  3  | test_onrpg  |   RPG  | fe8b103 |
| test_ppo | test_ppo.json |  3  | test_ppo  |   RPG  | fe8b103 |

  - Goal: 
    - test PPO with: state_normalizer, clip_ratio, gradient_clip
    - test OnRPG with: adv, adv_normalize, state_normalizer, actor_select, clip_ratio, gradient_clip
  - Analysis:
    - PPO: clip_ratio helps a lot; state_normalizer and gradient_clip also help a little bit;
    - OnRPG: gradient_clip helps a lot; clip_ratio is not that helpful; adv, adv_normalize, state_normalizer, and actor_select are still not clear.
  - Next: test OffRPG with new baseline; test OnRPG with different ways of clipping ratio; normalize predicted reward


| experiment | config file | runs |  log   | branch | commit  |
| ---------- | ----------- | ---- | ------ | ------ | ------- |
| test_onrpg1 | test_onrpg1.json |  3  | test_onrpg1  |   RPG  | fe8b103 |
| test_onrpg2 | test_onrpg2.json |  3  | test_onrpg2  |   RPG  | fe8b103 |
| test_onrpg3 | test_onrpg3.json |  3  | test_onrpg3  |   RPG  | fe8b103 |
| test_offrpg | test_offrpg.json |  1  | test_offrpg  |   RPG  | fe8b103 |

- Goal: 
  - test OnRPG with a new way of ratio clipping, different ways to normalize reward
  - test OffRPG with the new baseline V, different ways to normalize reward
- Analysis:
  - test_onrpg1: 
    - the new way of ratio clipping helps a little bit; 
    - gradient clipping helps
  - test_onrpg2:
    - use lambda_return helps; 
    - not clear for normalizing reward;
    - OnRPG1 is slightly better than OnRPG
  - test_onrpg3: 
    - normalize reward doesn't help
  - test_offrpg: 
    - still, MLPSquashedGaussianActor leads to NaN error; 
    - normalize reward doesn't help a lot
- Next: benchmark PPO and OnRPG on 6 env, 10 runs.
  - OnRPG: test on more environments with lambda_return
  - OffRPG: try MLPStdGaussianActor, test on more envs.


## 2021-01-09

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|   rpg_ppo  |  10  |   RPG  | 8698f73 |
| rpg_onrpg  |  10  |   RPG  | 8698f73 |
| rpg_offrpg |   5  |   RPG  | 8698f73 |

  - Goal: benchmark PPO, OnRPG, and OffRPG
  - Analysis: PPO is still better than OnRPG1 generally
    - OnRPG: OnRPG1 is better than OnRPG; lambda_return is better than v_next; gradient clipping helps.
    - PPO: gradient clipping helps.
    - OffRPG: bad; much worse than DDPG
  - Next: try new ways to normalize advantage


## 2021-01-10

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
| rpg_onrpg1 |  10  |   RPG  | 8698f73 |
|    ppo2    |  10  |   RPG  | 8698f73 |
  
  - Goal: try a new way to normalize reward; try the spinning up version of PPO
  - Analysis:
    - ppo2: PPO2 is better than PPO in most envs, but takes twice the training time of PPO.
    - rpg_onrpg1: the new way to normalize reward doesn't help in most envs; however, the new baseline V / gamma does help
  - Next: consider using state normalizer, gradient clipping, and the new baseline V / gamma.


## 2021-01-12

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
| rpg_onrpg  |  10  |   RPG  | a5fa26f |
|  rpg_ppo   |  10  |   RPG  | a5fa26f |

  - Goal: benchmark OnRPG and PPO with state_normalizer and gradient clip.
  - Analysis: 
    - OnRPG1 is consistently better than OnRPG; gradient_clipis is very helpful; state_normalizer helps in most envs.
    - OnRPG1 vs PPO: OnRPG1 wins 2 envs, losses 1, and 3 ties
  - Next: normalize adv with global std, i.e. std(reward+adv).


| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
| rpg_onrpg1 |  10  |   RPG  | a5fa26f |

  - Goal: normalize adv with global std in OnRPG1
  - Analysis: worse than OnRPG1 without normalizing adv.


## 2021-01-18

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
| rpg_onrpg2 |  10  |   RPG  | 17ae0ca |

  - Goal: test a variant of OnRPG1 that two IS ratios, one for reward part, one for state value. We only set IS ratio for state value to 0.
  - Analysis: it is better than OnRPG1 on Walker2d, but worse on several other environments.
  - Next:


## 2021-01-20

**Change the name of OnRPG1 to RPG.**

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|   rpg_lr   |  10  |   RPG  | 3c8e907 |

  - Goal: sweep lr for the reward net in RPG
  - Analysis: no lr is consistently better than the original lr (1e-3)
  - Next: final benchmark RPG and PPO


## 2021-01-21

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|   rpg1     |  20  |   RPG  | 3c8e907 |

  - Goal: benchmark RPG and PPO
  - Analysis:
    - Train: 2 wins, 3 losses, 1 tie
    - Test: 3 wins, 1 loss, 2 ties


| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|  rpg_reg1  |  10  |   RPG  | 3c8e907 |

  - Goal: use weight_decay to make reward net smooth
  - Analysis: except Reacher, there is improvement in other 5 envs.


## 2021-01-23

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|  rpg_reg2  |  10  |   RPG  | 3c8e907 |
|  rpg_reg3  |  10  |   RPG  | 3c8e907 |
|  rpg_reg4  |  10  |   RPG  | 3c8e907 |

  - Goal: sweep weight_decay with a smaller interval
  - Analysis: no dominant weigt_decay but 2e-6 to 6e-6 seems to be good choices; all in all, the best weight decay is 4e-6.


## 2021-01-24

| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|  rpg_reg   |  30  |   RPG  | 3824f1f |


  - Goal: benchmark RPG (with weight decay 4e-6) and PPO
  - Analysis: PPO is better now somehow with more runs while RPG is worse :(
    - Train: 1 win, 2 losses, 3 ties
    - Test:  2 wins, 2 losses, 2 ties


| experiment | runs | branch | commit  |
| ---------- | ---- | ------ | ------- |
|    rpg     |  30  |   RPG  | 3824f1f |

  - Goal: benchmark RPG (with no weight decay) and PPO on 6 tasks.
  - Analysis:
    - Train: 1 win, 2 losses, 3 ties
    - Test:  3 wins, 2 losses, 1 tie


## 2021-01-27

|  experiment  |  runs  | branch | commit  |
| ------------ | ------ | ------ | ------- |
| rpg_gradclip |   10   |   RPG  | 3824f1f |

  - Goal: sweep grad_clip for PPO and RPG on Cedar.
  - Analysis: no dominant gradient clip.