from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore


from mushroom_rl.core import Logger
from mushroom_rl.utils.dataset import compute_episodes_length


from air_hockey_challenge.environments.planar.hit import AirHockeyHit
import numpy as np

if __name__ == '__main__':
    env = AirHockeyHit(moving_init=True)

    #hack into reward function to make our own
    def custom_reward(state, action, next_state, absorbing):

        return 1
    
    env.reward = custom_reward

    #TODO: add controls from keyboard
    # wasd for x,y movement

    

    env.reset()
    env.render()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        action = np.zeros(3)

        observation, reward, done, info = env.step(action)
        env.render()

        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        print("J: ", J, " R: ", R)
        steps += 1
        if done or steps > env.info.horizon:
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()