from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from crossworld.policies.action import Action
from crossworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerCoffeeButtonV1Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "mug_pos": obs[3:6],
            "unused_info": obs[6:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_effort"] = -1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"]
        pos_mug = o_d["mug_pos"] + np.array([0.0, 0.0, 0.01])

        if abs(pos_curr[0] - pos_mug[0]) > 0.02:
            return np.array([pos_mug[0], pos_curr[1], 0.28])
        else:
            return pos_curr + np.array([0.0, 0.1, 0.0])
