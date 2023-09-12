import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

import type_aliases_mod as type_aliases
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env import is_vecenv_wrapped
from dummy_vec_env_mod import DummyVecEnv
# from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

import matplotlib.pyplot as plt

# def calc_win_rate(ep_rewards):
#     epsiode_rewards  = ep_rewards
#     agent_win_rate = 0   
    
#     for num in episode_rewards:
#         if num > 0:
#             agent_win_rate += 1

#     agent_win_rate =  round(agent_win_rate / n_eval_episodes,2)
    
#     op_win_rate  =  n_eval_episodes - agent_win_rate
#     op_win_rate =  round(op_win_rate / n_eval_episodes,2)
    
#     print("agent_win_rate", agent_win_rate, "op_win_rate", op_win_rate)

def  evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    turn_episode_rewardsre: bool = False,
    warn: bool = True,
    verbose: bool = False
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_rewards_op = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_rewards_op = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        if verbose: print("===============  game:", episode_counts[0]+1,"/", episode_count_targets[0], '===============')
        action_list= []
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        if verbose:
            print("Agent action:", actions)
        action_list.append(actions)
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_rewards_op += -1*rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                if verbose:
                    if done:
                        print("=============== game finished ===============")
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_rewards_op.append(-1 *info["episode"]["r"])
                            if verbose: print("your reward", -1 *info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_rewards_op.append(-1*current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_rewards_op[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    ag_mean_reward = np.mean(episode_rewards)
    op_mean_reward = np.mean(episode_rewards_op)
    
    ag_std_reward = np.std(episode_rewards)
    std_reward = ag_std_reward
    op_std_reward = np.std(episode_rewards_op)
    
    agent_win_rate = 0   

    for num in episode_rewards:
        if num > 0:
            agent_win_rate += 1

    agent_win_rate =  agent_win_rate / n_eval_episodes
    op_win_rate  =  100 - agent_win_rate
    agent_win_rate =  round(agent_win_rate,2)
    op_win_rate =  round(op_win_rate,2)
    
    
    if verbose: print("agent_win_rate", agent_win_rate, "op_win_rate", op_win_rate)
    if verbose: print("agent_mean_reward", ag_mean_reward, "op_mean_reward", op_mean_reward)
    if verbose: print("agent_std", ag_std_reward, "op_std", op_std_reward)
    
    mean_reward = ag_mean_reward 
    
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return mean_reward,episode_rewards, episode_lengths, episode_rewards_op
    if verbose:
        plt.hist(episode_rewards, bins=30, alpha=0.5, label='Agent rewards', color='blue')
        plt.hist(episode_rewards_op, bins=30, alpha=0.5, label='Opponent rewards', color='green')

        # Calculate and display the mean of each distribution
        plt.axvline(ag_mean_reward, color='blue', linestyle='dashed', linewidth=2, label=f'Agent mean reward: {ag_mean_reward:.2f}')
        plt.axvline(op_mean_reward, color='green', linestyle='dashed', linewidth=2, label=f'Opponent mean reward: {op_mean_reward:.2f}')

        # Add labels and legend
        plt.xlabel('Reward')
        plt.ylabel('Frequency')
        plt.legend(loc='best')

        # Show the plot
        plt.title('Players Reward Distribution')
        plt.show()
        
    return mean_reward, std_reward
