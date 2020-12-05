import gym
import gym_pygame
import gym_minatar
from gym.wrappers.time_limit import TimeLimit

from envs.wrapper import *


def make_env(env_name, max_episode_steps, episode_life=True):
  env = gym.make(env_name)
  env_group_title = get_env_group_title(env)
  # print(env_group_title, env_name)
  if env_group_title == 'gym_minatar':
    env = make_minatar(env, max_episode_steps, scale=False)
    if len(env.observation_space.shape) == 3:
      env = TransposeImage(env)
  elif env_group_title == 'atari' and '-ram' in env_name:
    make_atari_ram(env, max_episode_steps, scale=True)
  elif env_group_title == 'atari':
    env = make_atari(env, max_episode_steps)
    env = ReturnWrapper(env)
    env = wrap_deepmind(env,
                        episode_life=episode_life,
                        clip_rewards=False,
                        frame_stack=False,
                        scale=False)
    if len(env.observation_space.shape) == 3:
      env = TransposeImage(env)
    env = FrameStack(env, 4)
  elif env_group_title in ['box2d', 'classic_control', 'gym_pygame']:
    if max_episode_steps > 0: # Set max episode steps
      env = TimeLimit(env.unwrapped, max_episode_steps)
  return env


def get_env_group_title(env):
  '''
  Return the group name the environment belongs to.
  Possible group name includes: 
    - gym: atari, algorithmic, classic_control, box2d, toy_text, mujoco, robotics, unittest
    - gym_ple 
    - gym_pygame
  '''
  # env_name = env.unwrapped.spec.id
  s = env.unwrapped.spec.entry_point
  if 'gym_ple' in s:        # e.g. 'gym_ple:PLEEnv'
    group_title = 'gym_ple'
  elif 'gym_pygame' in s:   # e.g. 'gym_pygame.envs:CatcherEnv'
    group_title = 'gym_pygame'
  elif 'gym_minatar' in s:  # e.g. 'gym_minatar.envs:BreakoutEnv'
    group_title = 'gym_minatar'
  elif 'gym' in s:          # e.g. 'gym.envs.classic_control:CartPoleEnv'
    group_title = s.split('.')[2].split(':')[0]
  else:
    group_title = None
  
  return group_title