"""
Example Usage:
python3 BC_defense_train.py > train_outputs/run_<whatever number or descriptor>.py

Highly recommended to redirect output since it'll save time on training. You can also
provide the -p/--policy argument, which tells the script to continue training a policy
that we have a checkpoint of. Otherwise, a policy is trained from scratch.

When a policy is trained from a checkpoint, `bc.reconstruct_policy(<file_path>)` is used.
It's much easier than saving and loading the policy's state_dict() since by saving just 
the policy, then reconstructing it will return the entire policy.

Argument #2
=============
    -> (-exclude_robot_obs): option to train without

"""
import argparse
import pickle
from pathlib import Path

import gym
import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data.types import Transitions

from imitation.util import logger as imit_logger

from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from util_networks import AirHockeyACPolicy

from imitation.util import logger as imit_logger

#ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--policy", type=str, default="")
parser.add_argument("-e", "--exclude_robot_obs"  ,type=bool, default=False)
args = parser.parse_args()


#LOAD TRAINING DATA
with open("training_data.pkl", "rb") as f:
    training_data = pickle.load(f)
    obs = training_data["obs"]
    actions = training_data["actions"].reshape(training_data["actions"].shape[0],-1)
    next_obs = training_data["next_obs"] 
    dones = training_data["dones"]
    info = training_data["info"]


#SETUP MDP
mdp = AirHockeyChallengeWrapper(env="3dof-hit")
obs_space = mdp.info.observation_space

'''
    BIG NOTES:
        -> wtf is this shit
        -> observation space make no sense
'''
obs_space.low[0:3] = np.array([-10,-10,-np.pi])
obs_space.high[0:3] = np.array([10,10,np.pi])
bc_obs_space = gym.spaces.Box(low=obs_space.low, high=obs_space.high, shape=obs_space.shape)

action_space = mdp.info.action_space

#SETUP EXPERT SUPERVISED DATA
jnt_range = mdp.base_env.env_info['robot']['robot_model'].jnt_range
vel_range = mdp.base_env.env_info['robot']['joint_vel_limit'].T * 0.95

low = np.hstack((jnt_range[:,0][:,np.newaxis], vel_range[:,0][:,np.newaxis])).T
high = np.hstack((jnt_range[:,1][:,np.newaxis], vel_range[:,1][:,np.newaxis])).T

bc_action_space = gym.spaces.Box(low=low, high=high, shape=(2,3))
transitions = Transitions(obs=obs, acts=actions, infos=info, next_obs=next_obs, dones=dones)

#SETUP POLICY NETWORK FOR TRAINING
if len(args.policy) != 0:
    custom_policy = bc.reconstruct_policy(args.policy)
else:
    custom_policy = AirHockeyACPolicy(bc_obs_space, bc_action_space, exclude_robot_obs=args.exclude_robot_obs)


#SETUP IMITATION LEARNING
rng = np.random.default_rng()
bc_trainer = bc.BC(
    observation_space=bc_obs_space,
    action_space=bc_action_space,
    demonstrations=transitions,
    rng=rng,
    batch_size=80_000,
    policy=custom_policy.cuda(),
    custom_logger=imit_logger.configure('tensorboard_hit_base/'),
    device='cuda',
    l2_weight=10,
    ent_weight=10,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs={'lr': 1e-5, 'amsgrad': False}
)

bc_trainer.train(n_epochs=1000,
                progress_bar=True,
                reset_tensorboard=True,
                log_interval=10)

#SAVE MODEL
policy_ckpt_dir = Path("./policy_ckpts/")
if not policy_ckpt_dir.exists():
    policy_ckpt_dir.mkdir(parents=True, exist_ok=False)
n_files = len([_ for _ in policy_ckpt_dir.iterdir()])
bc_trainer.save_policy(f"policy_ckpts/defense_bc_policy_{n_files + 1}.pt")
