# The purpose of this page is to collect all the required python files used to fulfill the project.
#  The sections do not appear in sequential order, but in the order of that chosen in the dissertation.
#  The first part relate to the project design and offer quick access to the corresponding files.
#  NOTE: please open in debugging mode and change JSON setting to "justMyCode": false. this will alllow
#  easy access to the files. The second part relate to the project implementation. ie, how to code was
#  used to generate the results and findings used in the project. Finally, the third part include demonstrations:
# the exact same code as in the implementation and results section but with the variables changed to smaller values 
# to demonstrate the code runs successfully.

#import all the required files and libraries

import numpy as np   
import texas_holdem_mod as texas_holdem
from rlcard.utils.utils import print_card as prnt_cd
from rlcard.utils.utils import print_card as prnt_cd
from env_checker_mod import check_env
from evaluation_mod import evaluate_policy
from callbacks_mod import EvalCallback
from callbacks_mod import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
try:
    from stable_baselines3.common.callbacks_mod import BaseCallback
except ModuleNotFoundError:
    from stable_baselines3.common.callbacks import BaseCallback
import os 
import matplotlib.pyplot as plt
from ppo import PPO
from a2c import A2C
from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
import random
from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
from classmaker import graph_metrics
from classmaker import metric_dicts
from injector import card_injector
from human_input import human_play

# create the filepaths to your own device for the trained models in the  zip files included.
import trained_models
import os
# might not work so placed in try/except. please specify the path to the model zip files if this happens. 
try:
    trained_models_path = os.path.abspath('trained_models')
    PPO72plus10defaultFalse_10 = os.path.join(trained_models_path + '\\PPO72+10defaultFalse_10')
    PO72plus10defaultTrue_10 = os.path.join(trained_models_path + '\\PPO72+10defaultTrue_10')
    A2C72plus10defaultFalse_10 = os.path.join(trained_models_path + '\\A2C72+10defaultFalse_10')
    PPOPIG10defaultTruePIG72_10 = os.path.join(trained_models_path + '\\PPOPIG10defaultTruePIG72_10')

except Exception as e:
    print(f"An error occurred: {e}")   

# -----------------------------------Design-----------------------------------------------------------
    # Section 7.2: Train convergence test
from training_covergence_test import train_convergence_search

#     Section 7.3: Observation space amendments
from obs_amendments_exp import obs_experiment

#     Section 7.5: Custom wrapper for HULH
# - below is the modified raw env wrapper, the parent class for the custom wrapper
from texas_holdem_mod import raw_env
# - below is the custom wrapper, called 'meta_wrapper' since it is the last one applied
from texas_holdem_mod import meta_wrapper

#     Section 7.6: Selfplay architecture
from self_play import extract_opt_act
from self_play import CustomLoggerCallback
from self_play import self_play

#     Section 7.7: Data storage and plotting
from classmaker import metric_dicts
from classmaker import graph_metrics

#     Section 7.8: hyperparameter tunning
from primary_hyp_tune import BatchMultiplier
from primary_hyp_tune import hyperparam_search

#     Section 7.9: PPO algorithm and modifications
from ppo import PPO
# - below provides access to the function that shows the random (aka dumbmode) modification in the 
# collect_rollouts class method.
from on_policy_algorithm_mod import OnPolicyAlgorithm 
# - below provides access to the modifcations that provide actiion masking
from torch_layers_mod import MlpExtractor

#     Section 7.10: Optimal action NE_tool
from optimal_actions import extract_opt_act
from injector import card_injector

#     Section 7.11 Human oppponent functionality and interface 
from human_input import human_play

# ---------------------------IMPLEMENTATION AND RESULTS-----------------------------------------------------------------------------
#     Section 7.2: Train convergence test
TCS = train_convergence_search()
TCS.run(total_timesteps = 40000)
TCS.print_graphs()            

# #     Section 8.1: hyperparameter tuning results
def primary_hyp_search_PPO():
    """
    Perform primaryhyperparameter search for PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='PPO', obs_type='72+', trial_train_steps =40000)
    hypsrch.init_trained_op(training_steps=20480)
    print(hypsrch.run(print_graphs=True, numb_trials=120))
primary_hyp_search_PPO()
def primary_hyp_search_A2C():
    """
    Perform priamry hyperparameter search experiments for an A2C model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='A2C', obs_type='72+', trial_train_steps =40000)
    hypsrch.init_trained_op(training_steps=20480)
    print(hypsrch.run(print_graphs=True, numb_trials=120))
primary_hyp_search_A2C()
def secondary_hyp_search_PPO():
    """
    Perform secondary hyperparameter search experiments for an PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypersearch = micro_hyperparam_search(callback = None, verbose = True, model_type = 'PPO', obs_type = '72+',trial_train_steps =40000)
    hypersearch.init_trained_op(training_steps=20480)
    hypersearch.run(print_graphs=True, numb_trials=20)
secondary_hyp_search_PPO()        
def secondary_hyp_search_A2C():
    """
    Perform secondary hyperparameter search experiments for an PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
     """
    hypersearch = micro_hyperparam_search(callback = None, verbose = True, model_type = 'A2C', obs_type = '72+',trial_train_steps =40000)
    hypersearch.init_trained_op(training_steps=20480)
    hypersearch.run(print_graphs=True, numb_trials=20)
secondary_hyp_search_A2C()    

#     Section 8.2: Selfplay results and graph plotting
def self_play_group(n_gens, learning_steps,n_eval_episodes, obs_type, key, model_type, default_params, info):
    """
    This function creates an instance of the self_play class with specified parameters,
    runs the self-play training and evaluation loop, and generates evaluation graphs.

    Args:
    - None

    Returns:
    - None
    """
    sp = self_play(n_gens, learning_steps,n_eval_episodes, obs_type, key, model_type, default_params, info)
    sp.run(False)
    sp.get_results(graphs=True)
# # Hyperparameter tuned PPO with 72+ observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=30720, n_eval_episodes =10000, obs_type='72+', tag=1, model='PPO',na_key = None, default_params=False, info= 'project_main')
# # Hyperparameter tuned PPO with default observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=30720, n_eval_episodes =10000, obs_type='72', tag=2, model='PPO',na_key = None, default_params=False, info= 'project_main')
# # Default Hyperparameter PPO with 72+ observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=30720, n_eval_episodes =10000, obs_type='72+', tag=3, model='PPO',na_key = None, default_params=False, info= 'project_main')
# 
# # Hyperparameter tuned A2C with 72+ observation space type selfplay trained for 10 generation
# self_play_group(n_gens=10, learning_steps=30720, n_eval_episodes =10000, obs_type='72+', tag=4, model='A2C',na_key = None, default_params=False, info= 'project_main')
# # Hyperparameter tuned A2C with default observation space type selfplay trained for 10 generation
# self_play_group(n_gens=10, learning_steps=30720, n_eval_episodes =10000, obs_type='72', tag=5, model='A2C',na_key = None, default_params=False, info= 'project_main')
# # Default Hyperparameter A2C with 72+ observation space type selfplay trained for 10 generation
# self_play_group(n_gens=10, learning_steps=30720, n_eval_episodes =10000, obs_type='72+', tag=6, model='A2C',na_key = None, default_params=False, info= 'project_main')      


# #     Section 8.3: Observation space amendments results
def default_obs_PPO_vs_random(PPO_path, eval_steps):
    PPO_vs_random = PPO_vs_OPPONENT( '72', 'random')
    PPO_vs_random.init_eval_env()
    PPO_vs_random.load_params_from_file(PPO_path, None)
    PPO_vs_random.load_metric_dicts_storage()
    PPO_vs_random.evaluate(eval_steps)
    PPO_vs_random.get_results(True)
    print(PPO_vs_random.mean_reward)
default_obs_PPO_vs_random(PPO72plus10defaultFalse_10 , 10000)
# result = 1.03
def amended_obs_PPO_vs_random(PPO_path, eval_steps):
    PPO_vs_random = PPO_vs_OPPONENT( '72+', 'random')
    PPO_vs_random.init_eval_env()
    PPO_vs_random.load_params_from_file(PPO_path, None)
    PPO_vs_random.load_metric_dicts_storage()
    PPO_vs_random.evaluate(eval_steps)
    PPO_vs_random.get_results(True)
    print(PPO_vs_random.mean_reward)
default_obs_PPO_vs_random(PPO72plus10defaultTrue_10 , 10000)
# result = 2.12

# #  Section 8.5: Nash Equilibrium analysis
# from NE_analysis import NE_tool
NET = NE_tool('72+', 10)
NET.run( n_eval_episodes=100)
NET.print_graph()

# # Section 8.6: KL divergence
from KL_div import kl_div_test
kl = kl_div_test('72+',10)
print(kl.run())

# # Section 8.7: Action optimality
# # graph returned from the previously run code for selfplay below.
self_play_group(n_gens=10, learning_steps=30720,n_eval_episodes = 10000, obs_type='72+', tag=7, model='PPO',na_key = None, default_params=False, info= 'project_main')

# # Section: 8.8, 8.9, 8.10, 8.11, 8.12 (inclusive perfomance evaluation against all opponents)

from PPO_vs_all_opponents import PPO_vs_allops
allops = PPO_vs_allops(eval_steps=10000, eval_steps_human=50)      
allops.run()

# #Section 8.13: Failed attempts (IGG agent vs PIG agent)
# # Hyperparameter tuned PPO with Perfect infromation game (PIG) observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=30720,n_eval_episodes = 10000, obs_type='PIG', tag=8, model='PPO', na_key = None, default_params=False, info= 'project_main')
def PIG_vs_random(eval_steps):
    PPO_path = PPOPIG10defaultTruePIG72_10 
    PIG_vs_random = PPO_vs_OPPONENT('PIG', 'random')
    PIG_vs_random.init_eval_env()
    PIG_vs_random.load_params_from_file(PPO_path, None)
    PIG_vs_random.load_metric_dicts_storage()
    PIG_vs_random.evaluate(eval_steps)
    PIG_vs_random.get_results(True)
    print(PIG_vs_random.mean_reward)
PIG_vs_random(10000)
# # result = 2.04

# -----------------------------------------demonstrations----------------------------------------------------
    # Section 7.2: Train convergence test
TCS = train_convergence_search()
TCS.run(total_timesteps = 40000)
TCS.print_graphs()            


#     Section 8.1: hyperparameter tuning results DEMO
def primary_hyp_search_PPO():
    """
    Perform primaryhyperparameter search for PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='PPO', obs_type='72+', trial_train_steps =4096)
    hypsrch.init_trained_op(training_steps=4096)
    print(hypsrch.run(print_graphs=True, numb_trials=5))
primary_hyp_search_PPO()
def primary_hyp_search_A2C():
    """
    Perform priamry hyperparameter search experiments for an A2C model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypsrch = hyperparam_search(callback=None, verbose=True, batch_size=[32, 64, 128, 256, 512, 1024], model_type='A2C', obs_type='72+', trial_train_steps =4096)
    hypsrch.init_trained_op(training_steps=4096)
    print(hypsrch.run(print_graphs=True, numb_trials=5))
primary_hyp_search_A2C()
def secondary_hyp_search_PPO():
    """
    Perform secondary hyperparameter search experiments for an PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
    """
    hypersearch = micro_hyperparam_search(callback = None, verbose = True, model_type = 'PPO', obs_type = '72+',trial_train_steps =4096)
    hypersearch.init_trained_op(training_steps=4096)
    hypersearch.run(print_graphs=True, numb_trials=2)
secondary_hyp_search_PPO()        
def secondary_hyp_search_A2C():
    """
    Perform secondary hyperparameter search experiments for an PPO model and print results.

    This function initializes a hyperparameter search instance with specified parameters,
    conducts a hyperparameter search experiment, and prints the results including graphs.

    Args:
    - None

    Returns:
    - None
     """
    hypersearch = micro_hyperparam_search(callback = None, verbose = True, model_type = 'A2C', obs_type = '72+',trial_train_steps =4096)
    hypersearch.init_trained_op(training_steps=4096)
    hypersearch.run(print_graphs=True, numb_trials=2)
secondary_hyp_search_A2C()       

#     Section 8.2: Selfplay results and graph plotting
def self_play_group(n_gens, learning_steps,n_eval_episodes, obs_type, tag, model, na_key, default_params, info):

    """
    This function creates an instance of the self_play class with specified parameters,
    runs the self-play training and evaluation loop, and generates evaluation graphs.

    Args:
    - None

    Returns:
    - None

    """

    
    sp = self_play(n_gens, learning_steps, n_eval_episodes, obs_type, tag, model, na_key, default_params, info)
    sp.run(False)
    sp.get_results(graphs=True)

# # Hyperparameter tuned PPO with 72+ observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72+', tag=9, model='PPO',na_key = None, default_params=False, info= 'project_main')
# # Hyperparameter tuned PPO with default observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72', tag=10, model='PPO',na_key = None, default_params=False, info= 'project_main')
# # Default Hyperparameter PPO with 72+ observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72+', tag=11, model='PPO',na_key = None, default_params=False, info= 'project_main')
# 
# # Hyperparameter tuned A2C with 72+ observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72+', tag=12, model='A2C',na_key = None, default_params=False, info= 'project_main')
# # Hyperparameter tuned A2C with default observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72', tag=13, model='A2C',na_key = None, default_params=False, info= 'project_main')
# # Default Hyperparameter A2C with 72+ observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72+', tag=14, model='A2C',na_key = None, default_params=False, info= 'project_main') 

# #     Section 8.3: Observation space amendments results

from PPO_vs_opponent import PPO_vs_OPPONENT
def default_obs_PPO_vs_random(PPO_path, eval_steps):
    PPO_vs_random = PPO_vs_OPPONENT( '72', 'random')
    PPO_vs_random.init_eval_env()
    PPO_vs_random.load_params_from_file(PPO_path , None)
    PPO_vs_random.load_metric_dicts_storage()
    PPO_vs_random.evaluate(eval_steps)
    PPO_vs_random.get_results(True)
    print(PPO_vs_random.mean_reward)


default_obs_PPO_vs_random(PPO72plus10defaultTrue_10, 10)

def amended_obs_PPO_vs_random(PPO_path, eval_steps):
    PPO_vs_random = PPO_vs_OPPONENT( '72+', 'random')
    PPO_vs_random.init_eval_env()
    PPO_vs_random.load_params_from_file(PPO_path, None)
    PPO_vs_random.load_metric_dicts_storage()
    PPO_vs_random.evaluate(eval_steps)
    PPO_vs_random.get_results(True)
    print(PPO_vs_random.mean_reward)
default_obs_PPO_vs_random(PPO72plus10defaultFalse_10, 10000)


# #  Section 8.5: Nash Equilibrium analysis
# from NE_analysis import NE_tool
NET = NE_tool('72+', 10)
NET.run(n_eval_episodes=100)
NET.print_graph()

# # Section 8.6: KL divergence
from KL_div.py import kl_div_test
kl = kl_div_test('72+',10)
print(kl.run())

# # Section 8.7: Action optimality
# # graph returned from the previously run code for selfplay below.
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='72+', tag=15, model='PPO',na_key = None, default_params=False, info= 'project_main')

# # Section: 8.8, 8.9, 8.10, 8.11, 8.12 (inclusive perfomance evaluation against all opponents)
from PPO_vs_all_opponents import PPO_vs_allops
allops = PPO_vs_allops(eval_steps = 100, eval_steps_human=15)      
allops.run()

# #Section 8.13: Failed attempts (IGG agent vs PIG agent)

# # Hyperparameter tuned PPO with Perfect infromation game (PIG) observation space type selfplay trained for 10 generation
self_play_group(n_gens=10, learning_steps=6144, n_eval_episodes =1000, obs_type='PIG', tag=16, model='PPO',na_key = None, default_params=False, info= 'project_main')
def PIG_vs_random_default_obs(eval_steps=10):
    PPO_path = PPOPIG10defaultTruePIG72_10 
    PIG_vs_random = PPO_vs_OPPONENT('PIG', 'random')
    PIG_vs_random.init_eval_env()
    PIG_vs_random.load_params_from_file(PPO_path, None)
    PIG_vs_random.load_metric_dicts_storage()
    PIG_vs_random.evaluate(eval_steps)
    PIG_vs_random.get_results(True)
    print(PIG_vs_random.mean_reward)
PIG_vs_random_default_obs()






