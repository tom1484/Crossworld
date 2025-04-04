from __future__ import annotations

import copy
import pickle
from typing import Any, Callable, Literal, SupportsFloat
from abc import ABC, abstractmethod

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.envs.mujoco import MujocoEnv as mjenv_gym
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle
from typing_extensions import TypeAlias

from metaworld.types import XYZ, EnvironmentStateDict, ObservationDict, Task


RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"


class MocapBase(mjenv_gym, ABC):
    """Provides some commonly-shared functions for Arm Mujoco envs that use mocap for XYZ control."""

    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 80,
    }

    def __init__(
        self,
        model_name: str,
        frame_skip: int = 5,
        render_mode: RenderMode | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ) -> None:
        mjenv_gym.__init__(
            self,
            model_name,
            frame_skip=frame_skip,
            observation_space=self.arm_observation_space,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )
        self.reset_mocap_welds()
        self.frame_skip = frame_skip

    def get_env_state(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Get the environment state.

        Returns:
            A tuple of (qpos, qvel).
        """
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        return copy.deepcopy((qpos, qvel))

    def set_env_state(
        self, state: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
    ) -> None:
        """
        Set the environment state.

        Args:
            state: A tuple of (qpos, qvel).
        """
        mocap_pos, mocap_quat = state
        self.set_state(mocap_pos, mocap_quat)

    def __getstate__(self) -> EnvironmentStateDict:
        """Returns the full state of the environment as a dict.

        Returns:
            A dictionary containing the env state from the `__dict__` method, the model name (path) and the mocap state `(qpos, qvel)`.
        """
        state = self.__dict__.copy()
        return {"state": state, "mjb": self.model_name, "mocap": self.get_env_state()}

    def __setstate__(self, state: EnvironmentStateDict) -> None:
        """Sets the state of the environment from a dict exported through `__getstate__()`.

        Args:
            state: A dictionary containing the env state from the `__dict__` method, the model name (path) and the mocap state `(qpos, qvel)`.
        """
        self.__dict__ = state["state"]
        mjenv_gym.__init__(
            self,
            state["mjb"],
            frame_skip=self.frame_skip,
            observation_space=self.arm_observation_space,
        )
        self.set_env_state(state["mocap"])

    ##############################
    # START Arm specific methods #
    ##############################

    @abstractmethod
    def reset_mocap_welds(self) -> None:
        """Resets the mocap welds that we use for actuation."""
        # if self.model.nmocap > 0 and self.model.eq_data is not None:
        #     for i in range(self.model.eq_data.shape[0]):
        #         if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
        #             self.model.eq_data[i] = np.array(
        #                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 5.0]
        #             )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The name of the model to load."""

    @property
    @abstractmethod
    def tcp_center(self) -> npt.NDArray[Any]:
        """The COM of the gripper's 2 fingers.

        Returns:
            3-element position.
        """

    @abstractmethod
    def get_endeff_pos(self) -> npt.NDArray[Any]:
        """Returns the position of the end effector."""

    @property
    @abstractmethod
    def arm_observation_space(self) -> Box:
        """Returns the observation space for the arm.

        Returns:
            gym.spaces.Box: The observation space for the arm
        """


class ArmEnv(MocapBase, EzPickle):
    _HAND_SPACE = None
    """Bounds for hand position."""

    max_path_length = 600
    """The maximum path length for the environment (the task horizon)."""

    TARGET_RADIUS = 0.05
    """Upper bound for distance from the target when checking for task completion."""

    class _Decorators:
        @classmethod
        def assert_task_is_set(cls, func: Callable) -> Callable:
            """Asserts that the task has been set in the environment before proceeding with the function call.
            To be used as a decorator for SawyerXYZEnv methods."""

            def inner(*args, **kwargs) -> Any:
                env = args[0]
                if not env._set_task_called:
                    raise RuntimeError(
                        "You must call env.set_task before using env." + func.__name__
                    )
                return func(*args, **kwargs)

            return inner

    def __init__(
        self,
        frame_skip: int = 5,
        hand_low: XYZ = (-0.2, 0.55, 0.05),
        hand_high: XYZ = (0.2, 0.75, 0.3),
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        action_scale: float = 1.0 / 100,
        action_rot_scale: float = 1.0,
        render_mode: RenderMode | None = None,
        camera_id: int | None = None,
        camera_name: str | None = None,
        done_on_success: bool = False,
        ignore_termination: bool = False,
    ):
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)

        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        
        self.done_on_success = done_on_success
        # For policy testing and debug purpose
        self.ignore_termination = ignore_termination

        self.curr_path_length: int = 0
        self.seeded_rand_vec: bool = False
        self._freeze_rand_vec: bool = True
        self._last_rand_vec: npt.NDArray[Any] | None = None
        self.num_resets: int = 0
        self.current_seed: int | None = None
        self.obj_init_pos: npt.NDArray[Any] | None = None

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space: Box | None = None
        self.discrete_goals: list = []
        self.active_discrete_goal: int | None = None

        self._partially_observable: bool = True

        super().__init__(
            self.model_name,
            frame_skip=frame_skip,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        mujoco.mj_forward(
            self.model, self.data
        )  # *** DO NOT REMOVE: EZPICKLE WON'T WORK *** #

        self._did_see_sim_exception: bool = False

        self.action_space = None  # OVERRIDE ME

        # Technically these observation lengths are different between v1 and v2,
        # but we handle that elsewhere and just stick with v2 numbers here
        self._obs_obj_max_len: int = 14
        self._set_task_called: bool = False
        # self.done_on_success = False

        self.hand_init_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._target_pos: npt.NDArray[Any] | None = None  # OVERRIDE ME
        self._random_reset_space: Box | None = None  # OVERRIDE ME
        self.goal_space: Box | None = None  # OVERRIDE ME
        self._last_stable_obs: npt.NDArray[np.float64] | None = None

        # Note: It is unlikely that the positions and orientations stored
        # in this initiation of _prev_obs are correct. That being said, it
        # doesn't seem to matter (it will only effect frame-stacking for the
        # very first observation)

        self.init_qpos = np.copy(self.data.qpos)
        self.init_qvel = np.copy(self.data.qvel)
        self._prev_obs = self._get_curr_obs_combined_no_goal()

        EzPickle.__init__(
            self,
            self.model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            action_scale,
            action_rot_scale,
        )

    def seed(self, seed: int) -> list[int]:
        """Seeds the environment.

        Args:
            seed: The seed to use.

        Returns:
            The seed used inside a 1 element list.
        """
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        assert self.goal_space
        self.goal_space.seed(seed)
        return [seed]

    def set_task(self, task: Task) -> None:
        """Sets the environment's task.

        Args:
            task: The task to set.
        """
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data["env_cls"])
        del data["env_cls"]
        self._freeze_rand_vec = True
        self._last_rand_vec = data["rand_vec"]
        del data["rand_vec"]
        self._partially_observable = data["partially_observable"]
        del data["partially_observable"]
        self._set_task_inner(**data)

    def discretize_goal_space(self, goals: list) -> None:
        """Discretizes the goal space into a Discrete space.
        Current disabled and callign it will stop execution.

        Args:
            goals: List of goals to discretize
        """
        assert False, "Discretization is not supported at the moment."
        assert len(goals) >= 1
        self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    def _get_site_pos(self, site_name: str) -> npt.NDArray[np.float64]:
        """Gets the position of a given site.

        Args:
            site_name: The name of the site to get the position of.

        Returns:
            Flat, 3 element array indicating site's location.
        """
        # WARNING: Not sure what this is doing
        # _id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        # return self.data.site_xpos[_id].copy()
        return self.data.site(site_name).xpos.copy()

    def _set_pos_site(self, name: str, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the site corresponding to `name`.

        Args:
            name (str): The site's name
            pos (np.ndarray): Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1

        # _id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        # self.data.site_xpos[_id] = pos[:3]
        self.data.site(name).xpos = pos[:3]

    @property
    def touching_main_object(self) -> bool:
        """Calls `touching_object` for the ID of the env's main object.

        Returns:
            Whether the gripper is touching the object
        """
        return self.touching_object(self._get_id_main_object())

    def _get_pos_goal(self) -> npt.NDArray[Any]:
        """Retrieves goal position from mujoco properties or instance vars.

        Returns:
            Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:
        """Combines the end effector's {pos, closed amount} and the object(s)' {pos, quat} into a single flat observation.

        Note: The goal's position is *not* included in this.

        Returns:
            The flat observation array (18 elements)
        """

        pos_hand = self.get_endeff_pos()

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        return np.hstack((pos_hand, self.gripper_distance_apart, obs_obj_padded))

    def _get_obs(self) -> npt.NDArray[np.float64]:
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the goal position to form a single flat observation.

        Returns:
            The flat observation array (39 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def _get_obs_dict(self) -> ObservationDict:
        obs = self._get_obs()
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[3:-3],
        )

    @_Decorators.assert_task_is_set
    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], SupportsFloat, bool, bool, dict[str, Any]]:
        """Step the environment.

        Args:
            action: The action to take. Must be a 4 element array of floats.

        Returns:
            The (next_obs, reward, terminated, truncated, info) tuple.
        """
        assert len(action) == 4, f"Actions should be size 4, got {len(action)}"
        self.set_xyz_action(action[:3])

        if not self.ignore_termination and self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            assert self._last_stable_obs is not None
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,
                False,  # termination flag always False
                {  # info
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )

        mujoco.mj_forward(self.model, self.data)
        self._last_stable_obs = self._get_obs()

        self._last_stable_obs = np.clip(
            self._last_stable_obs,
            a_max=self.arm_observation_space.high,
            a_min=self.arm_observation_space.low,
            dtype=np.float64,
        )
        assert isinstance(self._last_stable_obs, np.ndarray)
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        # step will never return a terminate==True if there is a success
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True

        done = False
        if info["success"] and self.done_on_success:
            done = True
        
        truncate = truncate and not self.ignore_termination
        done = done and not self.ignore_termination

        return (
            np.array(self._last_stable_obs, dtype=np.float64),
            reward,
            done,
            truncate,
            info,
        )

    def reset_model(self) -> npt.NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
        """Resets the environment.

        Args:
            seed: The seed to use. Ignored, use `seed()` instead.
            options: Additional options to pass to the environment. Ignored.

        Returns:
            The `(obs, info)` tuple.
        """
        self.curr_path_length = 0
        self.reset_model()
        obs, info = super().reset()

        obs_dim = self._prev_obs.shape[0]
        self._prev_obs = obs[:obs_dim].copy()
        obs[obs_dim : obs_dim + obs_dim] = self._prev_obs
        obs = obs.astype(np.float64)
        return obs, info

    def _reset_hand(self, steps: int = 1) -> None:
        """Resets the hand position.

        Args:
            steps: The number of steps to take to reset the hand.
        """
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

    def _get_state_rand_vec(self) -> npt.NDArray[np.float64]:
        """Gets or generates a random vector for the hand position at reset."""
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        elif self.seeded_rand_vec:
            assert self._random_reset_space is not None
            rand_vec = self.np_random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            )
            self._last_rand_vec = rand_vec
            return rand_vec
        else:
            assert self._random_reset_space is not None
            rand_vec: npt.NDArray[np.float64] = np.random.uniform(  # type: ignore
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            ).astype(np.float64)
            self._last_rand_vec = rand_vec
            return rand_vec

    ##############################
    # START Arm specific methods #
    ##############################

    @abstractmethod
    def _gripper_caging_reward(
        self,
        action: npt.NDArray[np.float32],
        obj_pos: npt.NDArray[Any],
        obj_radius: float,
        pad_success_thresh: float,
        object_reach_radius: float,
        xz_thresh: float,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
    ):
        """Reward for agent grasping obj.

        Args:
            action(np.ndarray): (4,) array representing the action
                delta(x), delta(y), delta(z), gripper_effort
            obj_pos(np.ndarray): (3,) array representing the obj x,y,z
            obj_radius(float):radius of object's bounding sphere
            pad_success_thresh(float): successful distance of gripper_pad
                to object
            object_reach_radius(float): successful distance of gripper center
                to the object.
            xz_thresh(float): successful distance of gripper in x_z axis to the
                object. Y axis not included since the caging function handles
                    successful grasping in the Y axis.
            desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
            high_density(bool): flag for high-density. Cannot be used with medium-density.
            medium_density(bool): flag for medium-density. Cannot be used with high-density.

        Returns:
            the reward value
        """

    @abstractmethod
    def set_xyz_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): The action to apply
        """

    @property
    @abstractmethod
    def gripper_distance_apart(self) -> float:
        """Gets the distance between the gripper pads.
        Returns:
            The distance between the gripper pads
        """

    @abstractmethod
    def touching_object(self, object_geom_id: int) -> bool:
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id (int): the ID of the object in question

        Returns:
            (bool): whether the gripper is touching the object

        """

    ##############################
    # MARK: Env specific methods #
    ##############################

    @staticmethod
    def _set_task_inner() -> None:
        """Helper method to set additional task data. To be overridden by subclasses as appropriate."""
        # Doesn't absorb "extra" kwargs, to ensure nothing's missed.
        return None

    def _set_obj_xyz(self, pos: npt.NDArray[Any]) -> None:
        """Sets the position of the object.

        Args:
            pos: The position to set as a numpy array of 3 elements (XYZ value).
        """
        arm_nqpos = self._HAND_SPACE.low.size
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # freejoint qpos: x, y, z qvel: vx, vy, vz, ax, ay, az
        qpos[arm_nqpos : arm_nqpos + 3] = pos.copy()
        qvel[arm_nqpos : arm_nqpos + 6] = 0
        self.set_state(qpos, qvel)

    @property
    def _target_site_config(self):
        """Retrieves site name(s) and position(s) corresponding to env targets.

        :rtype: list of (str, np.ndarray)
        """
        return [("goal", self._target_pos)]

    def _get_id_main_object(self) -> int:
        return self.data.geom("objGeom").id

    @abstractmethod
    def evaluate_state(
        self, obs: npt.NDArray[np.float64], action: npt.NDArray[np.float32]
    ) -> tuple[float, dict[str, Any]]:
        """Does the heavy-lifting for `step()` -- namely, calculating reward and populating the `info` dict with training metrics.

        Returns:
            float: Reward between 0 and 10
            dict: Dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)

        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it

    @abstractmethod
    def _get_pos_objects(self) -> npt.NDArray[Any]:
        """Retrieves object position(s) from mujoco properties or instance vars.

        Returns:
            Flat array (usually 3 elements) representing the object(s)' position(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it

    @abstractmethod
    def _get_quat_objects(self) -> npt.NDArray[Any]:
        """Retrieves object quaternion(s) from mujoco properties.

        Returns:
            Flat array (usually 4 elements) representing the object(s)' quaternion(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
