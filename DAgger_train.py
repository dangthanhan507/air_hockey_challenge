from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper


import gym
from gym import spaces

import numpy as np
import torch


'''
    CustomEnv:
        -> wraps our MushroomRL robot environment around OpenAI Gym Environment
        -> 
'''
class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mdp_name, arg2):
    super(CustomEnv, self).__init__()
    '''
        Requirements for this to work:
            #in init
            -> implement self.action_space
            -> implement self.observation_space


            -> implement step()
            -> implement reset()
            -> implement render()
            -> implement close()
    '''

    #mdp_name = "3dof-hit"
    mdp = AirHockeyChallengeWrapper(env=mdp_name)

    #setup observation space
    bc_obs_space = mdp.info.observation_space    
    bc_obs_space.low[0:3] = np.array([-10,-10,-np.pi])
    bc_obs_space.high[0:3] = np.array([10,10,np.pi])

    bc_obs_space = gym.spaces.Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape)

    #setup action space
    jnt_range = mdp.base_env.env_info['robot']['robot_model'].jnt_range
    vel_range = mdp.base_env.env_info['robot']['joint_vel_limit'].T * 0.95

    low = np.hstack((jnt_range[:,0][:,np.newaxis], vel_range[:,0][:,np.newaxis])).T
    high = np.hstack((jnt_range[:,1][:,np.newaxis], vel_range[:,1][:,np.newaxis])).T
    bc_action_space = gym.spaces.Box(low=low, high=high, shape=(2,3))


    #save MushroomRL mdp
    self.mdp = mdp

    self.observation_space = bc_obs_space
    self.action_space = bc_action_space
    
  def step(self, action):
    return observation, reward, done, info
  def reset(self):
    return self.mdp.reset()

  def render(self, mode='human'):
    print('Hello world')
    print('\t why you print my like that? :<')
    self.mdp.render()
    pass
  
  def close (self):
