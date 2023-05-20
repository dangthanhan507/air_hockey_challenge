import datetime
import gc
import itertools
import json
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from imitation.algorithms.bc import reconstruct_policy
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from mushroom_rl.core import Logger
from mushroom_rl.utils.dataset import compute_episodes_length

from air_hockey_challenge.framework import AgentBase
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import \
    AirHockeyChallengeWrapper
from air_hockey_challenge.framework.challenge_core import ChallengeCore

PENALTY_POINTS = {"joint_pos_constr": 2, "ee_constr": 3, "joint_vel_constr": 1, "jerk": 1, "computation_time_minor": 0.5,
                  "computation_time_middle": 1,  "computation_time_major": 2}

def custom_evaluate(agent_builder, log_dir, env_list, n_episodes=1080, n_cores=-1, seed=None, generate_score=None,
             quiet=True, render=False, maybe_plot=True, maybe_generate_trajs, **kwargs):
    """
    WILL ALSO RETURN A TRANSITIONS OBJECT

    Function that will run the evaluation of the agent for a given set of environments. The resulting Dataset and
    constraint stats will be written to folder specified in log_dir. The resulting Dataset can be replayed by the
    replay_dataset function in air_hockey_challenge/utils/replay_dataset.py. This function is intended to be called
    by run.py.

    Args:
        agent_builder ((mdp_info, **kwargs) -> Agent): Function that returns an agent given the env_info and **kwargs.
        log_dir (str): The path to the log directory
        env_list (list): List of environments, on which the agent is tested
        n_episodes (int, 1000): Number of episodes each environment is evaluated
        n_cores (int, -1): Number of parallel cores which are used for the computation. -1 Uses all cores.
            When using 1 core the program will not be parallelized (good for debugging)
        seed (int, None): Desired seed to be used. The seed will be set for numpy and torch.
        generate_score(str, None): Set to "phase-1" or "phase-2" to generate a report for the first or second phase.
            The report shows the performance stats which will be used in the scoring of the agent. Note that the
            env_list has to include ["3dof-hit-opponent", "3dof-defend"] to generate the phase-1 report and
            ["7dof-hit", "7dof-defend", "7dof-prepare"] for the phase-2 report.
        quiet (bool, True): set to True to disable tqdm progress bars
        render (bool, False): set to True to spawn a viewer that renders the simulation
        kwargs (any): Argument passed to the Agent init
    """
    print("=== CUSTOM EVALUATE ===")
    path = os.path.join(log_dir, datetime.datetime.now().strftime('eval-%Y-%m-%d_%H-%M-%S'))

    if n_cores == -1:
        n_cores = os.cpu_count()

    assert n_cores != 0
    assert n_episodes >= n_cores

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    results = {}
    env_init_chuncks = []

    # Precompute all initial states for all experiments
    for env in env_list:
        env_init_chuncks.append(generate_init_states(env, n_episodes, n_cores))

    summary = "=================================================\n"
    for env, chunks in zip(env_list, env_init_chuncks):

        # returns: dataset, success, penalty_sum, constraints_dict, jerk, computation_time, violations, metric_dict
        data = Parallel(n_jobs=n_cores)(delayed(_evaluate)(path, env, agent_builder, chunks[i], quiet, render,
                                                           sum([len(x) for x in chunks[:i]]), compute_seed(seed, i), i, maybe_plot, maybe_generate_trajs,
                                                           **kwargs) for i in range(n_cores))
        print(f"DATA: {data}")

        logger = Logger(log_name=env, results_dir=path)

        success = np.mean([d[0] for d in data])
        penalty_sum = np.sum([d[1] for d in data])
        violations = data[0][2]
        for d in data[1:]:
            violations.update(d[2])

        constraint_names = data[0][3]
        n_steps = [d[4] for d in data]

        # Compute violation stats
        violation_stats = defaultdict(int)
        for eps in violations.values():
            for el in eps:
                if "jerk" in el:
                    violation_stats["Jerk"] += 1

                if "computation" in el:
                    violation_stats["Computation Time"] += 1

                for name in constraint_names:
                    if name in el:
                        violation_stats[name] += 1

        violation_stats["Total"] = sum(violation_stats.values())

        violations["violation summary"] = violation_stats

        with open(os.path.join(logger.path, "violations.json"), 'w') as f:
            json.dump(violations, f, indent=4)

        del violations

        # Concat the data saved by the workers. One by one so its memory efficient
        computation_time = np.zeros(sum(n_steps))
        for i in range(n_cores):
            computation_time[sum(n_steps[:i]): sum(n_steps[:i+1])] = np.load(os.path.join(logger.path, f"computation_time-{i}.npy"))
            os.remove(os.path.join(logger.path, f"computation_time-{i}.npy"))

        logger.log_numpy_array(computation_time=computation_time)
        del computation_time

        jerk_0 = np.load(os.path.join(logger.path, "jerk-0.npy"))
        os.remove(os.path.join(logger.path, "jerk-0.npy"))
        jerk = np.zeros((sum(n_steps), jerk_0.shape[1]))
        jerk[:n_steps[0]] = jerk_0
        for i in range(1, n_cores):
            jerk[sum(n_steps[:i]): sum(n_steps[:i + 1])] = np.load(
                os.path.join(logger.path, f"jerk-{i}.npy"))
            os.remove(os.path.join(logger.path, f"jerk-{i}.npy"))

        logger.log_numpy_array(jerk=jerk)
        del jerk
        del jerk_0

        for name in constraint_names:
            const_0 = np.load(os.path.join(logger.path, f"{name}-0.npy"))
            os.remove(os.path.join(logger.path, f"{name}-0.npy"))
            const = np.zeros((sum(n_steps), const_0.shape[1]))
            const[:n_steps[0]] = const_0
            for i in range(1, n_cores):
                const[sum(n_steps[:i]): sum(n_steps[:i + 1])] = np.load(
                    os.path.join(logger.path, f"{name}-{i}.npy"))
                os.remove(os.path.join(logger.path, f"{name}-{i}.npy"))

            logger.log_numpy_array(**{name: const})

        del const
        del const_0

        dataset = []
        for i in range(n_cores):
            dataset.extend(np.load(os.path.join(logger.path, f"dataset-{i}.pkl"), allow_pickle=True))
            os.remove(os.path.join(logger.path, f"dataset-{i}.pkl"))

        logger.log_dataset(dataset)

        del dataset

        gc.collect()

        color = 2
        if penalty_sum <= 500:
            color = 0
        elif penalty_sum <= 1500:
            color = 1

        results[env] = {"Success": success, "Penalty": penalty_sum, "Color": color}

        summary += "Environment: ".ljust(20) + f"{env}\n" + "Number of Episodes: ".ljust(20) + f"{n_episodes}\n"
        summary += "Success: ".ljust(20) + f"{success:.4f}\n"
        summary += "Penalty: ".ljust(20) + f"{penalty_sum}\n"
        summary += "Number of Violations: \n"
        for key, value in violation_stats.items():
            summary += f"  {key}".ljust(20) + f"{value}\n"
        summary += "-------------------------------------------------\n"

    print(summary)
    if generate_score:
        weights = defaultdict(float)
        if generate_score == "phase-1":
            weights.update({"3dof-hit": 0.5, "3dof-defend": 0.5})

        elif generate_score == "phase-2":
            weights.update({"7dof-hit": 0.4, "7dof-defend": 0.4, "7dof-prepare": 0.2})

        max_penalty_env = env_list[np.argmax([results[env]["Penalty"] for env in env_list])]
        score = {"Score": sum([results[env]["Success"] * weights[env] for env in env_list]),
                 "Max Penalty": results[max_penalty_env]["Penalty"],
                 "Color": results[max_penalty_env]["Color"]}

        results["Total Score"] = score

        with open(os.path.join(path, "results.json"), 'w') as f:
            json.dump(results, f, indent=4, default=convert)

        os.system("chmod -R 777 {}".format(log_dir))


def evaluate(agent_builder, log_dir, env_list, n_episodes=1080, n_cores=-1, seed=None, generate_score=None,
             quiet=True, render=False, maybe_plot=False, maybe_generate_trajs=False, **kwargs):
    """
    Function that will run the evaluation of the agent for a given set of environments. The resulting Dataset and
    constraint stats will be written to folder specified in log_dir. The resulting Dataset can be replayed by the
    replay_dataset function in air_hockey_challenge/utils/replay_dataset.py. This function is intended to be called
    by run.py.

    Args:
        agent_builder ((mdp_info, **kwargs) -> Agent): Function that returns an agent given the env_info and **kwargs.
        log_dir (str): The path to the log directory
        env_list (list): List of environments, on which the agent is tested
        n_episodes (int, 1000): Number of episodes each environment is evaluated
        n_cores (int, -1): Number of parallel cores which are used for the computation. -1 Uses all cores.
            When using 1 core the program will not be parallelized (good for debugging)
        seed (int, None): Desired seed to be used. The seed will be set for numpy and torch.
        generate_score(str, None): Set to "phase-1" or "phase-2" to generate a report for the first or second phase.
            The report shows the performance stats which will be used in the scoring of the agent. Note that the
            env_list has to include ["3dof-hit-opponent", "3dof-defend"] to generate the phase-1 report and
            ["7dof-hit", "7dof-defend", "7dof-prepare"] for the phase-2 report.
        quiet (bool, True): set to True to disable tqdm progress bars
        render (bool, False): set to True to spawn a viewer that renders the simulation
        kwargs (any): Argument passed to the Agent init
    """

    path = os.path.join(log_dir, datetime.datetime.now().strftime('eval-%Y-%m-%d_%H-%M-%S'))

    if n_cores == -1:
        n_cores = os.cpu_count()

    assert n_cores != 0
    assert n_episodes >= n_cores

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    results = {}
    env_init_chuncks = []

    # Precompute all initial states for all experiments
    for env in env_list:
        env_init_chuncks.append(generate_init_states(env, n_episodes, n_cores))

    summary = "=================================================\n"
    for env, chunks in zip(env_list, env_init_chuncks):

        # returns: dataset, success, penalty_sum, constraints_dict, jerk, computation_time, violations, metric_dict
        data = Parallel(n_jobs=n_cores)(delayed(_evaluate)(path, env, agent_builder, chunks[i], quiet, render,
                                                           sum([len(x) for x in chunks[:i]]), compute_seed(seed, i), i, maybe_plot, maybe_generate_trajs,
                                                           **kwargs) for i in range(n_cores))

        logger = Logger(log_name=env, results_dir=path)

        success = np.mean([d[0] for d in data])
        penalty_sum = np.sum([d[1] for d in data])
        violations = data[0][2]
        for d in data[1:]:
            violations.update(d[2])

        constraint_names = data[0][3]
        n_steps = [d[4] for d in data]

        # Compute violation stats
        violation_stats = defaultdict(int)
        for eps in violations.values():
            for el in eps:
                if "jerk" in el:
                    violation_stats["Jerk"] += 1

                if "computation" in el:
                    violation_stats["Computation Time"] += 1

                for name in constraint_names:
                    if name in el:
                        violation_stats[name] += 1

        violation_stats["Total"] = sum(violation_stats.values())

        violations["violation summary"] = violation_stats

        with open(os.path.join(logger.path, "violations.json"), 'w') as f:
            json.dump(violations, f, indent=4)

        del violations

        # Concat the data saved by the workers. One by one so its memory efficient
        computation_time = np.zeros(sum(n_steps))
        for i in range(n_cores):
            computation_time[sum(n_steps[:i]): sum(n_steps[:i+1])] = np.load(os.path.join(logger.path, f"computation_time-{i}.npy"))
            os.remove(os.path.join(logger.path, f"computation_time-{i}.npy"))

        logger.log_numpy_array(computation_time=computation_time)
        del computation_time

        jerk_0 = np.load(os.path.join(logger.path, "jerk-0.npy"))
        os.remove(os.path.join(logger.path, "jerk-0.npy"))
        jerk = np.zeros((sum(n_steps), jerk_0.shape[1]))
        jerk[:n_steps[0]] = jerk_0
        for i in range(1, n_cores):
            jerk[sum(n_steps[:i]): sum(n_steps[:i + 1])] = np.load(
                os.path.join(logger.path, f"jerk-{i}.npy"))
            os.remove(os.path.join(logger.path, f"jerk-{i}.npy"))

        logger.log_numpy_array(jerk=jerk)
        del jerk
        del jerk_0

        for name in constraint_names:
            const_0 = np.load(os.path.join(logger.path, f"{name}-0.npy"))
            os.remove(os.path.join(logger.path, f"{name}-0.npy"))
            const = np.zeros((sum(n_steps), const_0.shape[1]))
            const[:n_steps[0]] = const_0
            for i in range(1, n_cores):
                const[sum(n_steps[:i]): sum(n_steps[:i + 1])] = np.load(
                    os.path.join(logger.path, f"{name}-{i}.npy"))
                os.remove(os.path.join(logger.path, f"{name}-{i}.npy"))

            logger.log_numpy_array(**{name: const})

        del const
        del const_0

        dataset = []
        for i in range(n_cores):
            dataset.extend(np.load(os.path.join(logger.path, f"dataset-{i}.pkl"), allow_pickle=True))
            os.remove(os.path.join(logger.path, f"dataset-{i}.pkl"))

        logger.log_dataset(dataset)

        del dataset

        gc.collect()

        color = 2
        if penalty_sum <= 500:
            color = 0
        elif penalty_sum <= 1500:
            color = 1

        results[env] = {"Success": success, "Penalty": penalty_sum, "Color": color}

        summary += "Environment: ".ljust(20) + f"{env}\n" + "Number of Episodes: ".ljust(20) + f"{n_episodes}\n"
        summary += "Success: ".ljust(20) + f"{success:.4f}\n"
        summary += "Penalty: ".ljust(20) + f"{penalty_sum}\n"
        summary += "Number of Violations: \n"
        for key, value in violation_stats.items():
            summary += f"  {key}".ljust(20) + f"{value}\n"
        summary += "-------------------------------------------------\n"

    print(summary)
    if generate_score:
        weights = defaultdict(float)
        if generate_score == "phase-1":
            weights.update({"3dof-hit": 0.5, "3dof-defend": 0.5})

        elif generate_score == "phase-2":
            weights.update({"7dof-hit": 0.4, "7dof-defend": 0.4, "7dof-prepare": 0.2})

        max_penalty_env = env_list[np.argmax([results[env]["Penalty"] for env in env_list])]
        score = {"Score": sum([results[env]["Success"] * weights[env] for env in env_list]),
                 "Max Penalty": results[max_penalty_env]["Penalty"],
                 "Color": results[max_penalty_env]["Color"]}

        results["Total Score"] = score

        with open(os.path.join(path, "results.json"), 'w') as f:
            json.dump(results, f, indent=4, default=convert)

        os.system("chmod -R 777 {}".format(log_dir))

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

def build_bc_agent(env_info, **kwargs):
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

def _evaluate(log_dir, env, agent_builder, init_states, quiet, render, episode_offset, seed, i, maybe_plot, maybe_generate_trajs, **kwargs):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    eval_params = {"quiet": quiet, "render": render, "initial_states": init_states}

    mdp = AirHockeyChallengeWrapper(env)

    agent = agent_builder(mdp.env_info, **kwargs)
    bc_defense_policy = reconstruct_policy("./defense_bc_policy2_1024")
    device = torch.device("cpu")
    bc_agent = build_bc_agent(mdp.env_info, model=bc_defense_policy, device=device, **kwargs)
    core = ChallengeCore(agent, mdp)

    dataset, success, penalty_sum, constraints_dict, jerk, computation_time, violations = compute_metrics(core, bc_agent, eval_params, episode_offset, maybe_plot, maybe_generate_trajs)

    logger = Logger(log_name=env, results_dir=log_dir, seed=i)

    logger.log_dataset(dataset)
    logger.log_numpy_array(jerk=jerk, computation_time=computation_time, **constraints_dict)

    n_steps = len(computation_time)

    return success, penalty_sum, violations, constraints_dict.keys(), n_steps


def compute_metrics(core, bc_agent, eval_params, episode_offset, maybe_plot=False, maybe_generate_trajs=False):
    dataset, dataset_info = core.evaluate(**eval_params, get_env_info=True)
    # print(f"DATASET LEN: {len(dataset)}")
    
    episode_length = compute_episodes_length(dataset)
    n_episodes = len(episode_length)

    # Turn list of dicts to dict of lists
    constraints_dict = {k: [dic[k] for dic in dataset_info["constraints_value"]] for k in dataset_info["constraints_value"][0]}
    success = 0
    penalty_sum = 0
    current_idx = 0
    current_eps = episode_offset
    violations = defaultdict(list)

    # Iterate over episodes
    for episode_len in episode_length:
        # For plotting
        if maybe_plot:
            episode_data = dataset[current_eps:current_eps + episode_len]
            x, y, z = [], [], []
            x_vel, y_vel, z_vel = [], [], []

            bc_x, bc_y, bc_z = [], [], []
            bc_x_vel, bc_y_vel, bc_z_vel = [], [], []

            bc_agent.reset()
            for data in episode_data:
                action = data[1]
                x.append(action[0,0])
                y.append(action[0,1])
                z.append(action[0,2])

                x_vel.append(action[1,0])
                y_vel.append(action[1,1])
                z_vel.append(action[1,2])

                obs = data[0]
                action = bc_agent.draw_action(obs)
                bc_x.append(action[0,0])
                bc_y.append(action[0,1])
                bc_z.append(action[0,2])
                bc_x_vel.append(action[1,0])
                bc_y_vel.append(action[1,1])
                bc_z_vel.append(action[1,2])

            check_movement_conditions = [
                    np.var(np.array(x)) < 1e-5,
                    np.var(np.array(y)) < 1e-5,
                    np.var(np.array(z)) < 1e-5,
                    np.var(np.array(x_vel)) < 1e-5,
                    np.var(np.array(y_vel)) < 1e-5,
                    np.var(np.array(z_vel)) < 1e-5
                ]

            if False:
                print("BAD DATA!!!")
            else:
                with open(f"evaluate_plots/episode_{current_idx}.txt", "a") as f:
                    for i in range(episode_len):
                        # f.write(x[i] + " " + y[i] + " " + z[i] + " " + x_vel[i] + " " + y_vel[i] + " " + z_vel[i])
                        f.write(f"{x[i]} {y[i]} {z[i]} {x_vel[i]} {y_vel[i]} + {z_vel[i]}\n")
                 
                plt.subplot(231)
                plt.title("X")
                plt.plot(np.arange(episode_len), np.array(x))
                plt.plot(np.arange(episode_len), np.array(bc_x))
                plt.subplot(232)
                plt.title("Y")
                plt.plot(np.arange(episode_len), np.array(y))
                plt.plot(np.arange(episode_len), np.array(bc_y))
                plt.subplot(233)
                plt.title("Z")
                plt.plot(np.arange(episode_len), np.array(z))
                plt.plot(np.arange(episode_len), np.array(bc_z))
                plt.subplot(234)
                plt.title("X vel")
                plt.plot(np.arange(episode_len), np.array(x_vel))
                plt.plot(np.arange(episode_len), np.array(bc_x_vel))
                plt.subplot(235)
                plt.title("Y vel")
                plt.plot(np.arange(episode_len), np.array(y_vel))
                plt.plot(np.arange(episode_len), np.array(bc_y_vel))
                plt.subplot(236)
                plt.title("Z vel")
                plt.plot(np.arange(episode_len), np.array(z_vel))
                plt.plot(np.arange(episode_len), np.array(bc_z_vel))
                plt.savefig(f"evaluate_plots/episode_{current_idx}.jpg")

        # Only check last step of episode for success
        episode_result = dataset_info["success"][current_idx + episode_len - 1]
        success += episode_result        

        for name in constraints_dict.keys():
            if np.any(np.array(constraints_dict[name][current_idx: current_idx + episode_len]) > 0):
                penalty_sum += PENALTY_POINTS[name]
                violations["Episode " + str(current_eps)].append(name + " violated")

        # TODO set jerk limit to value that makes sense, idk whats good
        if np.any(np.array(dataset_info["jerk"][current_idx: current_idx + episode_len]) > 10000):
            penalty_sum += PENALTY_POINTS["jerk"]
            violations["Episode " + str(current_eps)].append("jerk > 10000")

        max_time_violations = np.max(dataset_info["computation_time"][current_idx: current_idx + episode_len])
        mean_time_violations = np.mean(dataset_info["computation_time"][current_idx: current_idx + episode_len])
        if max_time_violations > 0.2 or mean_time_violations > 0.02:
            penalty_sum += PENALTY_POINTS["computation_time_major"]
            if max_time_violations > 0.2:
                violations["Episode " + str(current_eps)].append("max computation_time > 0.2s")
            else:
                violations["Episode " + str(current_eps)].append("mean computation_time > 0.02s")

        elif max_time_violations > 0.1:
            penalty_sum += PENALTY_POINTS["computation_time_middle"]
            violations["Episode " + str(current_eps)].append("max computation_time > 0.1s")

        elif max_time_violations > 0.02:
            penalty_sum += PENALTY_POINTS["computation_time_minor"]
            violations["Episode " + str(current_eps)].append("max computation_time > 0.02s")

        current_idx += episode_len
        current_eps += 1

    if maybe_generate_trajs:
        states, actions, next_states, step_infos, dones = [], [], [], [], []
        for data in dataset:
            (state, action, _, next_state, _, last) = data 
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            dones.append(last)
        for i in range(len(states)):
            step_infos.append(dataset_info)
        log_training_data(states, actions, next_states, dones, step_infos)

    success = success / n_episodes

    return dataset, success, penalty_sum, constraints_dict, dataset_info["jerk"], dataset_info["computation_time"], violations

def log_training_data(obs, actions, next_actions, dones, info):
    with open("training_data.pkl", "wb") as f:
        data = {
                "obs": np.stack(obs),
                "actions": np.stack(actions),
                "next_obs": np.stack(next_actions),
                "dones": np.stack(dones),
                "info": np.stack(info)
        }
        pickle.dump(data, f)

def compute_seed(seed, i):
    if seed is not None:
        return seed + i
    return None


def convert(o):
    if isinstance(o, np.int_): return int(o)
    return o


def generate_init_states(env, n_episodes, n_parallel_cores):
    mdp = AirHockeyChallengeWrapper(env)
    init_states = []
    chunk_lens = [n_episodes // n_parallel_cores + int(x < n_episodes % n_parallel_cores) for x in
                  range(n_parallel_cores)]

    for chunk_len in chunk_lens:
        init_states_chunk = np.zeros((chunk_len, ) + mdp.info.observation_space.low.shape)
        for i in range(chunk_len):
            mdp.reset()
            init_states_chunk[i] = mdp.base_env._obs.copy()
        init_states.append(init_states_chunk)
    return init_states
