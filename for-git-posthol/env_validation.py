
import numpy as np   
from pettingzoo.classic.rlcard_envs import texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
from pettingzoo.classic.rlcard_envs import texas_holdem
import rlcard 
from rlcard.utils.utils import print_card as prnt_cd
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import DQN
from gymnasium import Env
import optuna
import gym
import numpy as np
import torch as th
from torch import nn
from tabulate import tabulate
import pandas as pd
from rlcard.agents.human_agents.nolimit_holdem_human_agent import HumanAgent
from classmaker import graph_metrics
from classmaker import obs_type_envs
from classmaker import metric_dicts

from injector import card_injector
from stable_baselines3.common.env_checker import check_env

env = texas_holdem.env('72+', render_mode='rgb_array')
env.AGENT.policy = 'random'
env.OPPONENT.policy = 'random'
# It will check your custom environment and output additional warnings if needed
check_env(env)