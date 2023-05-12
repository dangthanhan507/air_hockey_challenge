from air_hockey_challenge.framework.evaluate_agent import evaluate, custom_evaluate
from baseline.baseline_agent.baseline_agent import build_agent
import pickle


config = {'render': False, 'quiet': True, 'n_episodes': 10000, 'n_cores': -1, 'log_dir': 'logs', 'seed': None, 'generate_score': 'phase-1', 'env_list': ['3dof-hit']}
custom_evaluate(build_agent, **config)
