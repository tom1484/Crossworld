import numpy as np
from gymnasium.spaces import Box

from crossworld.envs.asset_path_utils import full_v1_path_for
from crossworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class SawyerPlateSlideBackEnv(SawyerXYZEnv):
    def __init__(self):
        goal_low = (-0.1, 0.6, 0.015)
        goal_high = (0.1, 0.6, 0.015)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.0, 0.85, 0.02)
        obj_high = (0.0, 0.85, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0.0, 0.85, 0.02], dtype=np.float32),
            "hand_init_pos": np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.0, 0.6, 0.015])
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
        return full_v1_path_for("sawyer_xyz/sawyer_plate_slide.xml")

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

    def _get_pos_objects(self):
        return self.data.get_geom_xpos("objGeom")

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:11] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.objHeight = self.data.get_geom_xpos("objGeom")[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos[:3]
            goal_pos = obj_pos[3:]
            self._target_pos = goal_pos

        self.sim.model.body_pos[self.model.body_name2id("cabinet")] = self.obj_init_pos
        self._set_obj_xyz(np.array([0, 0.2]))
        self.maxDist = np.linalg.norm(
            self.data.get_geom_xpos("objGeom")[:-1] - self._target_pos[:-1]
        )
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos(
            "rightEndEffector"
        ), self._get_site_pos("leftEndEffector")
        self.init_fingerCOM = (rightFinger + leftFinger) / 2

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos(
            "rightEndEffector"
        ), self._get_site_pos("leftEndEffector")
        fingerCOM = (rightFinger + leftFinger) / 2

        pullGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)

        pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                np.exp(-(pullDist**2) / c2) + np.exp(-(pullDist**2) / c3)
            )
            pullRew = max(pullRew, 0)
        else:
            pullRew = 0

        reward = -reachDist + pullRew

        return [reward, reachDist, pullDist]
