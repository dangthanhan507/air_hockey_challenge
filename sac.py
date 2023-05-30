#challenge core imports
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mushroom_rl.algorithms.actor_critic import SAC

#general imports
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import CustomChallengeCore
from air_hockey_challenge.utils.kinematics import forward_kinematics


def custom_reward_function(base_env, state, action, next_state, absorbing):
    reward_value = (state[0] - state[6]) * 2 + (1/abs(state[1] - state[7])) * 2 + absorbing * 3 + \
            (next_state[0] - next_state[6]) * 4 + (1/abs(next_state[0] - next_state[6]))
    return reward_value


def an_reward(base_env, state, action, next_state, absorbing):
    #state = observation
    puck_pos, puck_vel = base_env.get_puck(state)
    joint_pos, joint_vel = base_env.get_joints(state)

    next_joint_pos, next_joint_vel = base_env.get_joints(next_state)
    next_puck_pos, next_puck_vel = base_env.get_puck(next_state)


    #punish for puck getting closer to left side of table
    ee_pos, _ = forward_kinematics(base_env.env_info['robot']['robot_model'], base_env.env_info['robot']['robot_data'], joint_pos)
    #ee_pos in robot frame, translate to now be in table frame
    ee_pos[0] -= 1.51
    punish_goal = ee_pos[0]-puck_pos[0]


    next_ee_pos, _ = forward_kinematics(base_env.env_info['robot']['robot_model'], base_env.env_info['robot']['robot_data'], next_joint_pos)
    next_ee_pos[0] -= 1.51
    punish_goal += next_ee_pos[0]-next_puck_pos[0]

    #punish if it is out of constraint
    punish_constraint = 0
    for _,val in base_env.env_info['constraints'].fun(joint_pos,joint_vel).items():
        if np.any(val >= 0):
            punish_constraint += 9000
    for _,val in base_env.env_info['constraints'].fun(action[0,:], action[1,:]).items():
        if np.any(val >= 0):
            punish_constraint += 9000
    for _,val in base_env.env_info['constraints'].fun(next_joint_pos, next_joint_vel).items():
        if np.any(val >= 0):
            punish_constraint += 9000
    

    reward_value = 0
    reward_value -= punish_goal
    reward_value -= punish_constraint
    # reward_value -= punish_distance


    return reward_value


class ActionGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, layer_width, use_cuda = False, dropout=False, activation = nn.LeakyReLU(0.1) ):
        super().__init__()
        
        num_layers = 20
        
        input_dim = input_dim[0]
        output_dim = output_dim[0]
        
        layers = [nn.Linear(input_dim, layer_width), activation]
        for i in range(num_layers-1):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(activation)
        layers.append(nn.Linear(layer_width, output_dim))
        layers.append(activation)
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, obs):
        out = self.model(obs.float())
        return out

class CriticNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, layer_width,**kwargs):
        '''
            Arguments:
            ----------
                input_shape: (m,) wh#ere m is length of input
                output_shape: (m,n,..) where it represents shape of output
        '''
        super().__init__()
        
        input_dim = input_shape[0]
        output_dim = 1
        for i in output_shape: 
            output_dim *= i
        
        self.output_shape = output_shape
        activation = nn.LeakyReLU(0.1)
        
        if 'num_layers' in kwargs.keys():
            num_layers = kwargs['num_layers']
        else:
            num_layers = 20
        
        if 'layer_width' in kwargs.keys():
            layer_width = kwargs['layer_width']
        else:
            layer_width = 10
        
        
        layers = [nn.Linear(input_dim, layer_width), activation]
        for i in range(num_layers-1):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(activation)
        layers.append(nn.Linear(layer_width, output_dim))
        layers.append(activation)
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, state, action):
        state_action = torch.cat((state.float(), action.float()), dim=1)
        out = self.model(state_action.float()) #reshape into a tensor of classes representing pos/vel
            
        return out.T


if __name__ == '__main__':
    MAYBE_USE_CUDA = torch.cuda.is_available() 

    mdp = AirHockeyChallengeWrapper(env="3dof-hit", action_type="position-velocity", interpolation_order=3, custom_reward_function=an_reward, debug=True)
    # mdp.info.action_space.low = np.array([-2.967060, -1.8, -2.094395, -1.570796, -1.570796, -2.094395])
    # mdp.info.action_space.high = np.array([2.967060, 1.8, 2.094395, 1.570796, 1.570796, 2.094395])
    input_shape = mdp.info.observation_space.shape
    output_shape = (6,)

    layer_width = 10 
    actor_mu_params = dict(network=ActionGenerator,
                           layer_width=layer_width,
                           input_shape=input_shape,
                           output_shape=output_shape,
                           use_cuda=MAYBE_USE_CUDA)
    actor_sigma_params = dict(network=ActionGenerator,
                              layer_width=layer_width,
                              input_shape=input_shape,
                              output_shape=output_shape,
                              use_cuda=MAYBE_USE_CUDA)

    # actor network
    actor_optimizer = {
            "class": torch.optim.Adam,
            "params": {"lr": 0.1}
            }

    # critic network
    critic_input_shape = 12+6
    critic_params = dict(
            network=CriticNetwork,
            layer_width=layer_width,
            input_shape=(critic_input_shape,),
            optimizer={"class": optim.Adam, "params": {"lr": 0.1}},
            output_shape=(1,),
            loss=F.smooth_l1_loss,
            use_cuda=MAYBE_USE_CUDA
            )
    algorithm_params = {'mdp_info': mdp.info,
                        'actor_mu_params': actor_mu_params,
                        'actor_sigma_params': actor_sigma_params,
                    'actor_optimizer': actor_optimizer,
                    'critic_params': critic_params,
                    'batch_size': 16,
                    'initial_replay_size': 500,
                    'max_replay_size': 5000,
                    'warmup_transitions':100,
                    'tau': 0.001, #tau = soft update coefficient
                    'lr_alpha': 3e-4,
                   }
    sac = SAC(**algorithm_params)
    core = CustomChallengeCore(sac, mdp)
    core.learn(n_episodes=100, n_episodes_per_fit=100, render=True)

    with open("sac_weights.pkl", "wb") as f:
        pickle.dump(sac.policy.get_weights(), f)
