from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from crossworld.policies.action import Action
from crossworld.policies.policy import Policy, assert_fully_parsed, move


class PandaFaucetOpenV2Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "unused_gripper": obs[3],
            "faucet_pos": obs[4:7],
            "unused_info": obs[7:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_effort": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=25.0
        )
        action["grab_effort"] = 1.0

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        pos_curr = o_d["hand_pos"] + np.array([0.0, 0.01, 0.0])
        pos_faucet = o_d["faucet_pos"] + np.array([-0.04, 0.0, 0.05])
        pos_mid = pos_faucet + np.array([0.04, 0.02, 0.0])
        pos_final = pos_faucet + np.array([0.1, 0.06, 0.0])

        if np.linalg.norm(pos_curr[:2] - pos_faucet[:2]) > 0.05:
            return pos_faucet + np.array([0.0, 0.0, 0.1])
        elif abs(pos_curr[2] - pos_faucet[2]) > 0.05:
            return pos_faucet
        elif pos_mid[0] - pos_curr[0] > 0.01:
            return pos_mid
        else:
            return pos_final
