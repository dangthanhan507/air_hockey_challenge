import numpy as np
import torch
import tempfile
import pickle

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from pathlib import Path
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from util_networks import AirHockeyACPolicy

from openai_gym_custom import *

from stable_baselines3 import PPO

from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm


from imitation.data.types import Transitions

import argparse

'''
    Notes:
    =======
        -> Can swap PPO with SAC
        -> GAIL should be warm-started with Behavior Cloning / DAgger
        -> We are learning a reward network.
            -> play with structure of that network  
'''


if __name__ == '__main__':
    
    
    #ARGUMENTS
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--policy", type=str, default="")
    parser.add_argument("-e", "--exclude_robot_obs"  ,type=bool, default=False)
    args = parser.parse_args()
    
    
    env = CustomEnv('3dof-hit')

    venv = DummyVecEnv([lambda: CustomEnv('3dof-hit')])
    # venv = SubprocVecEnv([make_env() for i in range(3)])
    venv.reset()

    rng = np.random.default_rng(0)


    reward_net = BasicRewardNet(venv.observation_space,venv.action_space,normalize_input_layer=RunningNorm)

    expert = BaselineAgentPolicy(env.mdp.env_info, env.observation_space, env.action_space)

    with open("training_data.pkl", "rb") as f:
        training_data = pickle.load(f)
        obs = training_data["obs"]
        actions = training_data["actions"].reshape(training_data["actions"].shape[0],-1)
        next_obs = training_data["next_obs"] 
        dones = training_data["dones"]
        info = training_data["info"]

    transitions = Transitions(obs=obs, acts=actions, infos=info, next_obs=next_obs, dones=dones)

    custom_policy = AirHockeyACPolicy


    # full PPO tuning shit
    # learner = PPO(env=venv, 
    #               policy=custom_policy.cpu(),
    #               device='cpu',
    #               learning_rate=1e-3,
    #               n_steps=10000,
    #               batch_size=10_000,
    #               n_epochs=10_000,
    #               gamma=0.25,
    #               gae_lambda=0.5,
    #               normalize_advantage=True,
    #               ent_coef=10,
    #               vf_coef=10,
    #               clip_range=0.2,
    #               clip_range_vf=1)
    
    learner = PPO(policy=custom_policy, env=venv, n_steps=64,
                  policy_kwargs={'exclude_robot_obs': args.exclude_robot_obs, 'bigger_network': True, 'load_path': args.policy},
                  device='cuda:0')

    


    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=4096,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True
    )

    gail_trainer.train(20_000)

    #SAVE MODEL
    policy_ckpt_dir = Path("./gail_policy_chkpts/")
    if not policy_ckpt_dir.exists():
        policy_ckpt_dir.mkdir(parents=True, exist_ok=False)
    n_files = len([_ for _ in policy_ckpt_dir.iterdir()])
    torch.save(gail_trainer.policy,f"gail_policy_chkpts/hit_bc_policy_{n_files + 1}.pt")