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

def extract_opt_act(env):
    """
    Extracts and computes the percentages of occurrences of the optimal action '1' for both the agent (acts_ag) and the opponent (acts_op) 
    from the given environment (env). The percentages are computed as the ratio of the number of '1's to the total number of 
    elements, expressed as percentages.

    Parameters:
    - env (Environment): The environment object containing the action sequences for the agent and opponent.

    Returns:
    - percentages_ag (list of float): A list of percentages representing the occurrences of '1', reprenting the optimal action was taken,  in the agent's actions.
    - percentages_op (list of float): A list of percentages representing the occurrences of '1', reprenting the optimal action was taken,  in the opponent's actions.

    Note:
    - If there are no elements in either acts_ag or acts_op, the corresponding percentage will be set to 0.0.
    - The percentages are computed incrementally for each action, starting from the 21st action so the plotted graph is smoother since the percentage has stabalised after the first 300 points.
    """
    acts_ag = env.opt_acts_ag
    acts_op = env.opt_acts_op
    
    # Initialize lists to store percentages
    percentages_ag = []
    percentages_op = []
    
    # Calculate percentages for agent
    total_elements_ag = len(acts_ag)
    ones_count_ag = acts_ag.count(1)
    
    if total_elements_ag == 0:
        percentages_ag.append(0.0)
    else:
        for i, act in enumerate(acts_ag[500:], start=501):
            ones_count_ag = acts_ag[:i+1].count(1)
            percentage_ag = (ones_count_ag / (i + 1)) * 100
            percentages_ag.append(percentage_ag)
    
    # Calculate percentages for opponent
    total_elements_op = len(acts_op)
    ones_count_op = acts_op.count(1)
    
    if total_elements_op == 0:
        percentages_op.append(0.0)
    else:
        for i, act in enumerate(acts_op[20:], start=21):
            ones_count_op = acts_op[:i+1].count(1)
            percentage_op = (ones_count_op / (i + 1)) * 100
            percentages_op.append(percentage_op)
    
    return percentages_ag, percentages_op