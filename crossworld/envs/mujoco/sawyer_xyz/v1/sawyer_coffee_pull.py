import numpy as np
from gymnasium.spaces import Box

from crossworld.envs.asset_path_utils import full_v1_path_for
from crossworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class SawyerCoffeePullEnv(SawyerXYZEnv):
    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.75, 0.0)
        obj_high = (0.05, 0.8, 0.0)
        goal_low = (-0.1, 0.6, -0.001)
        goal_high = (0.1, 0.7, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_pos": np.array([0, 0.75, 0.0]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.0, 0.6, 0])
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
            dtype=np.float64,
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high), dtype=np.float64)

    @property
    def model_name(self):
        return full_v1_path_for("sawyer_xyz/sawyer_coffee.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pullDist = self.compute_reward(action, ob)

        info = {
            "reachDist": reachDist,
            "goalDist": pullDist,
            "epRew": reward,
            "pickRew": None,
            "success": float(pullDist <= 0.07),
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [("mug_goal", self._target_pos)]

    def _get_pos_objects(self):
        return self.data.get_geom_xpos("objGeom")

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com("obj")[:2] - self.data.get_geom_xpos("objGeom")[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1], self.get_body_com("obj")[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config["obj_init_pos"])
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.objHeight = self.data.get_geom_xpos("objGeom")[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            self._target_pos = np.concatenate(
                (goal_pos[-3:-1], [self.obj_init_pos[-1]])
            )
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            machine_pos = goal_pos[:3] - np.array([0, -0.15, -0.27])
            button_pos = machine_pos + np.array([0.0, -0.12, 0.05])
            self.sim.model.body_pos[
                self.model.body_name2id("coffee_machine")
            ] = machine_pos
            self.sim.model.body_pos[self.model.body_name2id("button")] = button_pos

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPullDist = np.linalg.norm(
            self.obj_init_pos[:2] - np.array(self._target_pos)[:2]
        )

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos(
            "rightEndEffector"
        ), self._get_site_pos("leftEndEffector")
        self.init_fingerCOM = (rightFinger + leftFinger) / 2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos(
            "rightEndEffector"
        ), self._get_site_pos("leftEndEffector")
        fingerCOM = (rightFinger + leftFinger) / 2

        goal = self._target_pos

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        assert np.all(goal == self._get_site_pos("mug_goal"))
        reachDist = np.linalg.norm(fingerCOM - objPos)
        pullDist = np.linalg.norm(objPos[:2] - goal[:2])
        reachRew = -reachDist
        reachDistxy = np.linalg.norm(
            np.concatenate((objPos[:-1], [self.init_fingerCOM[-1]])) - fingerCOM
        )

        if reachDistxy < 0.05:  # 0.02
            reachRew = -reachDist + 0.1
            if reachDist < 0.05:
                reachRew += max(actions[-1], 0) / 50
        else:
            reachRew = -reachDistxy

        if reachDist < 0.05:
            pullRew = 1000 * (self.maxPullDist - pullDist) + c1 * (
                np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
            )
            pullRew = max(pullRew, 0)
        else:
            pullRew = 0

        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]
