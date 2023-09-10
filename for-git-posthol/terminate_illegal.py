from __future__ import annotations

from pettingzoo.utils.env import ActionType, AECEnv, ObsType
from pettingzoo.utils.env_logger import EnvLogger
from pettingzoo.utils.wrappers.base import BaseWrapper


class TerminateIllegalWrapper(BaseWrapper):
    """This wrapper terminates the game with the current player losing in case of illegal values.

    Args:
        illegal_reward: number that is the value of the player making an illegal move.
    """

    def __init__(self, env: AECEnv, illegal_reward: float):
        super().__init__(env)
        self._illegal_value = illegal_reward
        self._prev_obs = None
        self.illegal_move_players = []

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self._terminated = False
        self._prev_obs = None
        super().reset(seed=seed, options=options)

    def observe(self, agent: str) -> ObsType | None:
        obs = super().observe(agent)
        if agent == self.agent_selection:
            self._prev_obs = obs
        return obs
    
    # def observe(self) -> ObsType | None:
    #     print("here in tiw")
    #     agent = self.learner
    #     obs = super().observe()
    #     if agent == self.agent_selection:
    #         self._prev_obs = obs
    #     return obs

    def step(self, action: ActionType) -> None:
        
        current_obs = self.observe(self.agent_selection)
        current_agent = self.agent_selection
        if self._prev_obs is None:
            self.observe(self.agent_selection)
        assert self._prev_obs
        assert (
            "action_mask" in self._prev_obs
        ), "action_mask must always be part of environment observation as an element in a dictionary observation to use the TerminateIllegalWrapper"
        _prev_action_mask = self._prev_obs["action_mask"]
        # print("prev action mask in termillegal", _prev_action_mask)
        # print("terminations", self.terminations)
        # print("terminated", self._terminated)
        # print("truncations", self.truncations)
        self._prev_obs = None
        if self._terminated and (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)  # pyright: ignore[reportGeneralTypeIssues]
        elif (
            not self.terminations[self.agent_selection]
            and not self.truncations[self.agent_selection]
            and not _prev_action_mask[action]
        ):
            EnvLogger.warn_on_illegal_move()
            print("for", action, 'prev action mask', _prev_action_mask, "by", self.agent_selection)
            print("current_obs: ", current_obs['action_mask'])
            print("potential illegal move")
            self._cumulative_rewards[self.agent_selection] = 0
            self.terminations = {d: True for d in self.agents}
            self.truncations = {d: True for d in self.agents}
            self._prev_obs = None
            self.rewards = {d: 0 for d in self.truncations}
            self.rewards[current_agent] = float(self._illegal_value)
            self._accumulate_rewards()
            self._deads_step_first()
            self._terminated = True
        else:
            super().step(action)

    def __str__(self) -> str:
        return str(self.env)
