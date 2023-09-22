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
from PPO_vs_opponent import PPO_vs_OPPONENT
class PPO_vs_allops():
    """
    Class to compare a PPO agent against various opponents and display results.

    This class allows you to compare the performance of a PPO agent against different types of opponents
    (e.g., random, human, A2C, heuristic) in a Texas Hold'em environment. It initializes the opponents,
    evaluates the agents, and displays the results in a bar chart.

    Args:
    - eval_steps (int): Number of evaluation steps/games.

    Attributes:
    - rewards (dict): Dictionary to store mean rewards for different opponents.
    - eval_steps (int): Number of evaluation steps/games.
    - eval_steps_human (int): Number of evaluation steps for human opponent.
    - PPO_path (str): Path to the PPO agent's model parameters file.

    Methods:
    - PPO_vs_random(): Compare PPO agent against a random opponent.
    - PPO_vs_human(): Compare PPO agent against a human opponent.
    - PPO_vs_a2c(): Compare PPO agent against an A2C opponent.
    - PPO_vs_heuristic(): Compare PPO agent against a heuristic opponent.
    - bar_chart(): Generate and display a bar chart of mean rewards.
    - run(): Run the experiments and display results.
    """
    def __init__(self, eval_steps, eval_steps_human ):
        self.rewards = {}
        self.eval_steps = eval_steps 
        self.eval_steps_human =eval_steps_human 
        self.PPO_path = r'S:\MSC_proj\models\PPO72+10defaultFalse_10'

    def PPO_vs_random(self):
        PPO_vs_random = PPO_vs_OPPONENT( '72+', 'random')
        PPO_vs_random.init_eval_env()
        PPO_vs_random.load_params_from_file(self.PPO_path, None)
        PPO_vs_random.load_metric_dicts_storage()
        PPO_vs_random.evaluate(self.eval_steps)
        PPO_vs_random.get_results(True)
        self.rewards['PPO_vs_random'] = PPO_vs_random.mean_reward
        print(PPO_vs_random.mean_reward)

    def PPO_vs_human(self): 
        PPO_vs_human = human_play('72+', self.eval_steps_human, self.PPO_path)
        PPO_vs_human.play()
        self.rewards['PPO_vs_human'] =  PPO_vs_human.mean_reward

    def PPO_vs_a2c(self):
        PPO_vs_a2c = PPO_vs_OPPONENT('72+', 'A2C')
        PPO_vs_a2c.init_eval_env()
        PPO_vs_a2c.load_params_from_file(self.PPO_path, r'S:\MSC_proj\models\A2C72+10defaultFalse_10')
        PPO_vs_a2c.load_metric_dicts_storage()
        PPO_vs_a2c.evaluate(self.eval_steps)
        PPO_vs_a2c.get_results(True)
        self.rewards['PPO_vs_A2C'] = PPO_vs_a2c.mean_reward

    def PPO_vs_heuristic(self):
        PPO_vs_heuristic = PPO_vs_OPPONENT('72+', 'heuristic')
        PPO_vs_heuristic.init_eval_env()
        PPO_vs_heuristic.load_params_from_file(self.PPO_path, None)
        PPO_vs_heuristic.load_metric_dicts_storage()
        PPO_vs_heuristic.evaluate(self.eval_steps)
        PPO_vs_heuristic.get_results(True)
        self.rewards['PPO_vs_heuristic'] = PPO_vs_heuristic.mean_reward

    def bar_chart(self):
        categories = list(self.rewards.keys())
        counts = list(self.rewards.values())
        colors = ['darkblue','mediumblue', 'blue', 'lightblue']
        plt.bar(categories, counts, color = colors)
        plt.xlabel('Opponents')
        plt.ylabel('PPO mean reward')
        plt.title('PPO mean reward vs opponent for ' + str(self.eval_steps) + ' games')

        # Display the chart
        plt.xticks(rotation=45) 
        plt.tight_layout()  
        plt.show()
        plt.savefig('S:\\MSC_proj\\plots')


    def run(self):
        self.PPO_vs_random()
        self.PPO_vs_a2c()
        self.PPO_vs_human()
        self.PPO_vs_heuristic()
        self.bar_chart()