import numpy as np
from gymnasium.spaces import Box

from crossworld.envs.asset_path_utils import full_v1_path_for
from crossworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class SawyerSoccerEnv(SawyerXYZEnv):
    def __init__(self):
        goal_low = (-0.1, 0.8, 0.03)
        goal_high = (0.1, 0.9, 0.03)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.03)
        obj_high = (0.1, 0.7, 0.03)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_pos": np.array([0, 0.6, 0.03]),
            "obj_init_angle": 0.3,
            "hand_init_pos": np.array([0.0, 0.6, 0.2]),
        }
        self.goal = np.array([0.0, 0.9, 0.03])
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
        return full_v1_path_for("sawyer_xyz/sawyer_soccer.xml")

    @SawyerXYZEnv._Decorators.assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pushDist = self.compute_reward(action, ob)

        info = {
            "reachDist": reachDist,
            "goalDist": pushDist,
            "epRew": reward,
            "pickRew": None,
            "success": float(pushDist <= 0.07),
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos("objGeom")

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com("obj")[:2] - self.data.get_geom_xpos("objGeom")[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1], self.data.get_geom_xpos("objGeom")[-1]]

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
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[
                self.model.body_name2id("goal_whole")
            ] = self._target_pos

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(
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
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos(
            "rightEndEffector"
        ), self._get_site_pos("leftEndEffector")
        fingerCOM = (rightFinger + leftFinger) / 2

        goal = self._target_pos

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        assert np.all(goal == self._get_site_pos("goal"))
        reachDist = np.linalg.norm(fingerCOM - objPos)
        pushDist = np.linalg.norm(objPos[:2] - goal[:2])
        reachRew = -reachDist

        def reachCompleted():
            return reachDist < 0.05

        self.reachCompleted = reachCompleted()
        if self.reachCompleted:
            pushRew = 1000 * (self.maxPushDist - pushDist) + c1 * (
                np.exp(-(pushDist**2) / c2) + np.exp(-(pushDist**2) / c3)
            )
            pushRew = max(pushRew, 0)
        else:
            pushRew = 0
        reward = reachRew + pushRew

        return [reward, reachDist, pushDist]
