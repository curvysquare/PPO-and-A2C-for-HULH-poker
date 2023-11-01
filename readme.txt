## Description:

The Proximal Policy Optimisation reinforcement (PPO) learning algorithm is applied to the zero-sum, two player poker game of Texas Limit Hold'em. Using a version of the game available through the PettingZoo library, this was modified from a multi-agent to a single agent environment such that the PPO algorithm imported from StableBaselines3 could be deployed. After an experiment was ran to determine the optimal observation space type for the algorithm, the primary and secondary hyperparameters were tuned and then the agent was trained through a process of self-play. This involves the agent essentially playing against a previous version of itself in order for the learn a superior strategy as the ability of the opponent increases through out the repeating process. The aim is for the agent to discover a Nash Equilibrium strategy: one that the agent has no incentive to deviate from to increase its payoff and constitutes an optimal strategy for the game. 

Two different forms of self-play approaches were tested, one where the agent starts with the same policy as its opponent and attempts to learn a superior one, and then one in which the policy of the agent is reset and it learns anew. The former of these self-play approaches proved to be highly successful, resulting in an agent who achieved a resource constrained equilibrium strategy that proved superior to all opponents it was tested against. This included a beginner level human opponent, a opponent using a heuristic based strategy and also the similar Advantage Actor-Critic algorithm which underwent the same extent of hyperparameter tuning and self-play training. The fact that the resultant policy of the trained PPO displayed convergence to an equilibrium strategy in its moving mean reward, achieved a difference in probability distributions to the previous version of itself equal to zero and the level of regret shown in the best response analysis all provide support for the interpretation that the agent achieved an effective resource constrained equilibrium that is proximal to Nash Equilibrium.


Results summary:


Intstallation and usage Instructions:

- dont install rlcard as this is already included with required modifications 
- made using VScode, suggested to run in this IDE to assist with compatibility
- filepaths to save figures and trained models will likely have to be replaced with a filepath to a folder on your own system. 
- the 'project_main' file collects all the required python files used to fulfill the project.
- when running for the first time. the 'path_gym' function of _patch_env will raise an exception saying the environment is not recognised as an open AI gymnasium environment.this is caused by the custom wrapper modifications making it not being recognised, despite being fully compatible. You can either delete this file on your device and replace it with 
the modified patch_gym.py file included in this folder. Alternativley you can easily modify the _patch_env function by hashing out all the code and just place 'return env'.

Required packages:

plotly
stable_baselines3 extra
optuna
pettingzoo
treys (poker hand evlauator ) https://github.com/ihendley/treys



Regards - Rhys 
