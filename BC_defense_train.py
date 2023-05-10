import pickle
from imitation.data.types import Transitions
import numpy as np
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
import gym
from imitation.algorithms import bc
import torch
from util_networks import AirHockeyACPolicy

with open("defend_training_data.pkl", "rb") as f:
    training_data = pickle.load(f)
    obs = training_data["obs"]
    actions = training_data["actions"].reshape(training_data["actions"].shape[0],-1)
    next_obs = training_data["next_obs"] # <- for next run, this will be renamed to next obs
    dones = training_data["dones"]
    info = training_data["info"]

mdp = AirHockeyChallengeWrapper(env="3dof-hit")
obs_space = mdp.info.observation_space
bc_obs_space = gym.spaces.Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape)

#this deceives u.... action_space from mdp only shows the torque limits
action_space = mdp.info.action_space

jnt_range = mdp.base_env.env_info['robot']['robot_model'].jnt_range
vel_range = mdp.base_env.env_info['robot']['joint_vel_limit'].T * 0.95

low = np.hstack((jnt_range[:,0][:,np.newaxis], vel_range[:,0][:,np.newaxis])).T
high = np.hstack((jnt_range[:,1][:,np.newaxis], vel_range[:,1][:,np.newaxis])).T

bc_action_space = gym.spaces.Box(low=low, high=high, shape=(2,3))
transitions = Transitions(obs=obs, acts=actions, infos=info, next_obs=next_obs, dones=dones)

custom_policy = AirHockeyACPolicy(bc_obs_space, bc_action_space) 

rng = np.random.default_rng()
bc_trainer = bc.BC(
    observation_space=bc_obs_space,
    action_space=bc_action_space,
    demonstrations=transitions,
    rng=rng,
    batch_size=256,
    policy=custom_policy
)

bc_trainer.train(n_epochs=200)
bc_trainer.save_policy("defense_bc_policy2_1024")
