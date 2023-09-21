# The purpose of this page is to collect all the required python files used to fulfill the project.
#  The sections do not appear in sequential order, but in the order of that chosen in the dissertation.
#  The first half relate to the project design and offer quick access to the corresponding files.
#  NOTE: please open in debugging mode and change JSON setting to "justMyCode": false. this will alllow
#  easy access to the files. The second half relate to the project implementation. ie, how to code was
#  used to generate the results and findings used in the project.

#import all the required files and libraries

# import numpy as np   
# import texas_holdem_mod as texas_holdem
# from rlcard.utils.utils import print_card as prnt_cd
# from rlcard.utils.utils import print_card as prnt_cd
# from env_checker_mod import check_env
# from evaluation_mod import evaluate_policy
# from callbacks_mod import EvalCallback
# from callbacks_mod import StopTrainingOnNoModelImprovement
# from stable_baselines3.common.monitor import Monitor
# try:
#     from stable_baselines3.common.callbacks_mod import BaseCallback
# except ModuleNotFoundError:
#     from stable_baselines3.common.callbacks import BaseCallback
# import os 
# import matplotlib.pyplot as plt
# from ppo import PPO
# from a2c import A2C
# from gymnasium import Env
# import optuna
# import gym
# import numpy as np
# import torch as th
# from torch import nn
# from tabulate import tabulate
# import pandas as pd
# import random
# from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
# from classmaker import graph_metrics
# from classmaker import metric_dicts
# from injector import card_injector
# from human_input import human_play


# Design:
    # Section 7.2: Train convergence test
from train_conv_search import train_convergence_search

#     Section 7.3: Observation space amendments
from obs_amendments_exp import obs_experiment

#     Section 7.5: Custom wrapper for HULH
# - below is the modified raw env wrapper, the parent class for the custom wrapper
from texas_holdem_mod import raw_env
# - below is the custom wrapper, called 'meta_wrapper' since it is the last one applied
from texas_holdem_mod import meta_wrapper

#     Section 7.6: Selfplay architecture
from self_play import extract_opt_act
from selfplay import CustomLoggerCallback
from self_play import self_play

#     Section 7.7: Data storage and plotting
from classmaker import metric_dicts
from classmaker import graph_metrics

#     Section 7.8: hyperparameter tunning
from primary_hyp_tuner import BatchMultiplier
from primary_hyp_tuner import hyperparam_search

#     Section 7.9: PPO algorithm and modifications
from ppo import PPO
# - below provides access to the function that shows the random (aka dumbmode) modification
from on_policy_algorithm_mod import collect_rollouts
# - below provides access to the modifcations that provide actiion masking
from torch_layers_mod import MlpExtractor

#     Section 7.10: Optimal action NE_tool
from optimal_actions import extract_opt_act

#     Section 7.11 Human oppponent functionality and interface 
from human_input import human_play_callback

# Implementation and results:
#     Section 7.2: Train convergence test
TCS = train_convergence_search(verbose = False, obs_type = '72+', trial_training_steps = 2048, cb_frequency = 2)
# TCS = train_convergence_search(verbose = False, obs_type = '72+', trial_training_steps = 40000, cb_frequency = 10000)
TCS.init_trained_op(training_steps =2048)
# TCS.run(print_graphs= False)
results = {'1000,60': 2.4341, '1000,60': 1.8836,  '4000,75': 2.1669, '500,75': 2.07295, '500,75': 2.00825, '2000,75': 1.9293, '2000,75': 1.92875, '500,60': 2.2359, '2000,75': 1.7597, '2000,75': 1.5784, '1000,75': 2.3316, '500,60': 1.2743}
TCS.plot_dictionary(results)
print(TCS.trial_results)

#     Section 8.1: hyperparameter tuning results
#     Section 8.2: Selfplay results and graph plotting
#     Section 8.3: Observation space amendments results
    