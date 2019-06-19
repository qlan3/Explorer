import gym
import gym_pygame
from envs.wrapper import *


def make_env(env_name, episode_life=True):
  env = gym.make(env_name).unwrapped
  env_group_title = get_env_group_title(env)
  # print('env_group_title:', env_group_title)
  if env_group_title == 'atari':
    env = make_atari(env_name)
    env = wrap_deepmind(env,
                        episode_life=episode_life,
                        clip_rewards=False,
                        frame_stack=False,
                        scale=False)
    if len(env.observation_space.shape) == 3:
      env = TransposeImage(env)
    env = FrameStack(env, 4)
  elif env_group_title == 'classic_control':
    pass
  elif env_group_title == 'gym_pygame':
    pass
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
  s = env.unwrapped.spec._entry_point
  if 'gym_ple' in s: # e.g. 'gym_ple:PLEEnv'
    group_title = 'gym_ple'
  elif 'gym_pygame' in s: # e.g. 'gym_pygame.envs:CatcherEnv'
    group_title = 'gym_pygame'
  elif 'gym' in s: # e.g. 'gym.envs.classic_control:CartPoleEnv'
    group_title = s.split('.')[2].split(':')[0]
  else:
    group_title = None
  
  return group_title