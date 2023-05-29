import numpy as np
import torch
from imitation.algorithms.bc import reconstruct_policy

from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import (ChallengeCore,
                                                           CustomChallengeCore)
from air_hockey_challenge.framework.custom_evaluate_agent import \
    custom_evaluate
from util_networks import AirHockeyACPolicy


class BehaviorCloneAgent(AgentBase):
    def __init__(self, env_info, model, device, **kwargs):
        super().__init__(env_info, **kwargs)
        self.new_start = True
        self.hold_position = None

        self.model = model
        self.device = device

    def reset(self):
        self.new_start = True
        self.hold_position = None

    def draw_action(self, observation):
        if self.new_start:
            self.new_start = False
            self.hold_position = self.get_joint_pos(observation)

            velocity = np.zeros_like(self.hold_position)
            action = np.vstack([self.hold_position, velocity])
        else:
            obs_tensor = torch.tensor(observation,dtype=torch.float).reshape(1,12).detach()
            action,_,_ = self.model(obs_tensor)
            action = action.reshape(2,3).detach().numpy()


        return action
def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    Args:
        env_info (dict): The environment information
        kwargs (any): Additionally setting from agent_config.yml
    Returns:
         (AgentBase) An instance of the Agent
    """

    return BehaviorCloneAgent(env_info, **kwargs)

model = reconstruct_policy("dagger_policy_ckpts/hit_bc_policy_3.pt").cpu()
custom_evaluate(build_agent, 'some_folder/', ['3dof-hit'], 5, 1, model=model, device='cpu', render=True, n_plot=2)
