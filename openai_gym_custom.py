import numpy as np
import gym
from gym import spaces
from baseline.baseline_agent.baseline_agent import BaselineAgent
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from imitation.policies.base import HardCodedPolicy

class CustomEnv(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, mdp_name):
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

    ultimate_scalar = 100
    bc_obs_space = mdp.info.observation_space    
    bc_obs_space.low[0:3] = -ultimate_scalar*np.ones(3,dtype=np.float32)
    bc_obs_space.high[0:3] = ultimate_scalar*np.ones(3,dtype=np.float32)

    bc_obs_space.low[3:6] = -ultimate_scalar*np.ones(3,dtype=np.float32)
    bc_obs_space.high[3:6] = ultimate_scalar*np.ones(3,dtype=np.float32)

    bc_obs_space.low[6:9] = -ultimate_scalar*np.ones(3,dtype=np.float32)
    bc_obs_space.high[6:9] = ultimate_scalar*np.ones(3,dtype=np.float32)

    bc_obs_space.low[9:12] = -ultimate_scalar*np.ones(3,dtype=np.float32)
    bc_obs_space.high[9:12] = ultimate_scalar*np.ones(3,dtype=np.float32)


    bc_obs_space = gym.spaces.Box(low=bc_obs_space.low, high=bc_obs_space.high, shape=bc_obs_space.shape)

    #setup action space
    jnt_range = mdp.base_env.env_info['robot']['robot_model'].jnt_range
    vel_range = mdp.base_env.env_info['robot']['joint_vel_limit'].T * 0.95

    low = np.hstack((jnt_range[:,0][:,np.newaxis], vel_range[:,0][:,np.newaxis])).T
    high = np.hstack((jnt_range[:,1][:,np.newaxis], vel_range[:,1][:,np.newaxis])).T

    #big note: torch log_prob/sampling does not support multi-dimensional
    #must keep action space to be single-dimensional
    bc_action_space = gym.spaces.Box(low=low.reshape(6), high=high.reshape(6), shape=(6,))


    #save MushroomRL mdp
    self.mdp = mdp

    self.observation_space = bc_obs_space
    self.action_space = bc_action_space
    
    self._episode_steps = 0

  def step(self, action):
    '''
        in mushroomRL terminology:
            -> next_state: observation idk why they call it state (can be confusing)
            -> reward: reward
            -> absorbing: out of bounds or not
            -> step_info: wtf is this
    '''

    '''
        OpenAI Gym's info:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of the task) is reached.
                In this case further step() calls could return undefined results.
            truncated (bool): whether a truncation condition outside the scope of the MDP is satisfied.
                Typically a timelimit, but could also be used to indicate agent physically going out of bounds.
                Can be used to end the episode prematurely before a `terminal state` is reached.
            info (dictionary): `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, or individual reward terms that are combined to produce the total reward.
                It also can contain information that distinguishes truncation and termination, however this is deprecated in favour
                of returning two booleans, and will be removed in a future version.
    '''
    next_state, reward, absorbing, step_info = self.mdp.step(action.reshape(2,3))

    reward = np.float32(reward)     

    self._episode_steps += 1

    last = not(self._episode_steps < self.mdp.info.horizon and not absorbing)

    #in case anything goes wrong first look at step_info being passed in
    truncated = False
    terminated = last
    observation = next_state
    info = step_info
    
    return observation, reward, terminated, info
  
  def reset(self):
    '''
        OPENAI INFO:
        Returns:
            observation (object): Observation of the initial state. This will be an element of :attr:`observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
    '''
    self._episode_steps = 0
    return self.mdp.reset()

  def render(self, mode='human'):
    print('Hello world')
    self.mdp.render()
    pass
  
  def close(self):
    pass


#converting baseline agent into hardcoded policy
class ZeroPolicy(HardCodedPolicy):
    """Returns constant zero action."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
    
class BaselineAgentPolicy(HardCodedPolicy):
    """Returns constant zero action."""
    def __init__(self, env_info, observation_space, action_space):
       '''

       '''
       super().__init__(observation_space, action_space)

       self.agent = BaselineAgent(env_info)


    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
       return self.agent.draw_action(obs).reshape(6)
       
       


def make_env():
    def _f():
        env = CustomEnv('3dof-hit')
        return env
    return _f