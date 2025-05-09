from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from crossworld.policies.action import Action
from crossworld.policies.policy import Policy, assert_fully_parsed, move


class JacoStickPushV1Policy(Policy):
    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs: npt.NDArray[np.float64]) -> dict[str, npt.NDArray[np.float64]]:
        return {
            "hand_pos": obs[:3],
            "stick_pos": obs[3:6],
            "obj_pos": obs[6:-3],
            "goal_pos": obs[-3:],
        }

    def get_action(self, obs: npt.NDArray[np.float64]) -> npt.NDArray[np.float32]:
        o_d = self._parse_obs(obs)

        action = Action({"delta_pos": np.arange(3), "grab_pow": 3})

        action["delta_pos"] = move(
            o_d["hand_pos"], to_xyz=self._desired_pos(o_d), p=10.0
        )
        action["grab_pow"] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d: dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[Any]:
        hand_pos = o_d["hand_pos"]
        stick_pos = o_d["stick_pos"] + np.array([-0.02, 0.0, 0.0])
        obj_pos = o_d["obj_pos"]
        goal_pos = o_d["goal_pos"]

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02:
            return stick_pos + np.array([0.0, 0.0, 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(hand_pos[2] - stick_pos[2]) > 0.05 and stick_pos[-1] < 0.03:
            return stick_pos + np.array([0.0, 0.0, 0.03])
        elif abs(obj_pos[2] + 0.05 - hand_pos[2]) > 0.01:
            return np.array([hand_pos[0], hand_pos[1], obj_pos[2] + 0.05])
        # Move to the goal
        else:
            return np.array([goal_pos[0], goal_pos[1], hand_pos[2]])

    @staticmethod
    def _grab_effort(o_d: dict[str, npt.NDArray[np.float64]]) -> float:
        hand_pos = o_d["hand_pos"]
        stick_pos = o_d["stick_pos"] + np.array([-0.02, 0.0, 0.0])

        if (
            np.linalg.norm(hand_pos[:2] - stick_pos[:2]) > 0.02
            or abs(hand_pos[2] - stick_pos[2]) > 0.1
        ):
            return 0.0
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 0.8
