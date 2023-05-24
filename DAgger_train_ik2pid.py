import tempfile
from pathlib import Path

import gym
import numpy as np
import torch
from gym import spaces
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.base import HardCodedPolicy
from imitation.util import logger as imit_logger
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from air_hockey_challenge.utils.kinematics import forward_kinematics
from baseline.baseline_agent.baseline_agent import BaselineAgent
from util_networks import AirHockeyACPolicy

'''
    CustomEnv:
        -> wraps our MushroomRL robot environment around OpenAI Gym Environment
        -> 
'''
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
    def __init__(self, env, observation_space, action_space):
       '''

       '''
       super().__init__(observation_space, action_space)

       self.env = env
       self.agent = BaselineAgent(env.env_info)


    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        chosen_action = self.agent.draw_action(obs).reshape(6)
        joint_pos, _ = self.env.get_joints(np.concatenate((np.zeros((6,)), chosen_action)))
        ee_pos, _ = forward_kinematics(self.env.env_info['robot']['robot_model'], self.env.env_info['robot']['robot_data'], joint_pos)
        ee_pos[0] -= 1.51
        chosen_action[0:3] = ee_pos
        return chosen_action
       


def make_env():
    def _f():
        env = CustomEnv('3dof-hit')
        return env
    return _f

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = CustomEnv('3dof-hit')

    venv = DummyVecEnv([lambda: CustomEnv('3dof-hit')])
    # venv = SubprocVecEnv([make_env() for i in range(3)])
    venv.reset()

    rng = np.random.default_rng(0)

    custom_policy = AirHockeyACPolicy(env.observation_space, env.action_space, exclude_robot_obs=False, bigger_network=True)

    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        rng=rng,
        policy=custom_policy.to(device),
        device=device,
        batch_size=1024,
        l2_weight=0.01,
        ent_weight=0.001,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs={'lr': 1e-3, 'amsgrad': False},
        custom_logger=imit_logger.configure('tensorboard_hit_base/'),
    )

    expert = BaselineAgentPolicy(env.mdp.base_env, env.observation_space, env.action_space)


    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(80_000)


        #SAVE MODEL
        policy_ckpt_dir = Path("./dagger_policy_ckpts/")
        if not policy_ckpt_dir.exists():
            policy_ckpt_dir.mkdir(parents=True, exist_ok=False)
        n_files = len([_ for _ in policy_ckpt_dir.iterdir()])
        dagger_trainer.save_policy(f"dagger_policy_ckpts/hit_bc_policy_{n_files + 1}.pt")
