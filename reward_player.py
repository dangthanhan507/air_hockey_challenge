from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore


from mushroom_rl.core import Logger
from mushroom_rl.utils.dataset import compute_episodes_length


from air_hockey_challenge.environments.planar.hit import AirHockeyHit
import numpy as np


from pynput import keyboard

from air_hockey_challenge.utils.kinematics import inverse_kinematics, forward_kinematics, jacobian

if __name__ == '__main__':

    def custom_reward_function(base_env, state, action, next_state, absorbing):
        reward_value = (state[0] - state[6]) * 2 + (1/abs(state[1] - state[7])) * 2 + absorbing * 3 + \
            (next_state[0] - next_state[6]) * 4 + (1/abs(next_state[0] - next_state[6]))
        return reward_value
    

    mdp = AirHockeyChallengeWrapper(env="3dof-defend", action_type="position-velocity", interpolation_order=3, custom_reward_function=custom_reward_function, debug=True)
    env = mdp.base_env
    env_info = mdp.env_info

    enable_keyboard = False

    action = np.zeros((2,3))
    observation = np.zeros(12) # first 6 is robot, last 6 is puck

    def on_press(key):
        global action
        global observation
        if enable_keyboard:

            try:
                if key.char == 'w' or key.char == 'a' or key.char == 's' or key.char == 'd':
                    # observation is given

                    puck_pos, puck_vel = env.get_puck(observation)
                    joint_pos, joint_vel = env.get_joints(observation)

                    #numpy array ee_pos
                    ee_pos, _ = forward_kinematics(env_info['robot']['robot_model'],env_info['robot']['robot_data'],joint_pos)
                    # print('\tq:', joint_pos)
                    # print('\tee_pos: ',ee_pos)

                    change = 0.05
                    if key.char == 'w':
                        delta_y = change
                    elif key.char == 's':
                        delta_y = -change
                    else:
                        delta_y = 0

                    if key.char == 'a':
                        delta_x = -change
                    elif key.char == 'd':
                        delta_x = change
                    else:
                        delta_x = 0

                    
                    new_ee = ee_pos + np.array([delta_x,delta_y,0])
                    # print('\tnew_ee:', new_ee)
                    success,new_q = inverse_kinematics(env_info['robot']['robot_model'], env_info['robot']['robot_data'], new_ee, initial_q = joint_pos)


                    # print('\tnew q:', new_q)
                    action = np.zeros((2,3))

                    if success:
                        action[0,:] = new_q
                    else:
                        action[0,:] = joint_pos


            except AttributeError:
                pass


    def on_release(key):
        if enable_keyboard:
            try:
                if key.char == 'w':
                    delta_y = 0
                elif key.char == 'a':
                    delta_y = 0
                elif key.char == 's':
                    delta_x = 0
                elif key.char == 'd':
                    delta_x = 0
            except AttributeError:
                pass

            if key == keyboard.Key.esc:
                # Stop listener
                print('--> Ending Keyboard')
                return False
    
    # with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
    #     listener.join()

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    
    listener.start()

    

    #TODO: add controls from keyboard
    # wasd for x,y movement



    env.reset()
    env.render()
    R = 0.
    J = 0.
    gamma = 1.
    steps = 0
    while True:
        observation, reward, done, info = env.step(action)
        enable_keyboard = True

        env.render()

        gamma *= env.info.gamma
        J += gamma * reward
        R += reward
        print('Reward:', reward)
        steps += 1

        if done or steps > env.info.horizon:
            R = 0.
            J = 0.
            gamma = 1.
            steps = 0
            env.reset()