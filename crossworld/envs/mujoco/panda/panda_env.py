from __future__ import annotations

from typing import Any, Literal

import mujoco
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box
from typing_extensions import TypeAlias

from crossworld.envs.mujoco.utils import reward_utils
from crossworld.types import XYZ
from crossworld.envs.mujoco.arm_env import ArmEnv


RenderMode: TypeAlias = "Literal['human', 'rgb_array', 'depth_array']"


class PandaEnv(ArmEnv):

    _INIT_HAND_QPOS = np.array(
        [
            1.51067,
            -1.57213,
            -1.37564,
            -2.53261,
            -1.37605,
            1.42732,
            1.7101,
            0.04,
            -0.04,
        ]
    )
    _HAND_QUAT = np.array([0.0, 0.0, -1.0, 0.0])

    _HAND_SPACE = Box(
        np.array([-0.525, 0.348, -0.0525]),
        np.array([+0.525, 1.025, 0.7]),
        dtype=np.float64,
    )
    """Bounds for hand position."""

    max_path_length: int = 500
    """The maximum path length for the environment (the task horizon)."""

    TARGET_RADIUS: float = 0.05
    """Upper bound for distance from the target when checking for task completion."""

    def __init__(
        self,
        hand_low: XYZ = (-0.2, 0.55, 0.05),
        hand_high: XYZ = (0.2, 0.75, 0.3),
        mocap_low: XYZ | None = None,
        mocap_high: XYZ | None = None,
        action_scale: float = 1.0 / 100,
        action_rot_scale: float = 1.0,
        render_mode: RenderMode | None = None,
        camera_id: int | None = None,
        camera_name: str | None = None,
        ignore_termination: bool = False,
    ) -> None:
        super().__init__(
            hand_low=hand_low,
            hand_high=hand_high,
            mocap_low=mocap_low,
            mocap_high=mocap_high,
            action_scale=action_scale,
            action_rot_scale=action_rot_scale,
            render_mode=render_mode,
            camera_id=camera_id,
            camera_name=camera_name,
            ignore_termination=ignore_termination,
        )

        self.init_left_pad: npt.NDArray[Any] = self.left_pad
        self.init_right_pad: npt.NDArray[Any] = self.right_pad

        self.action_space = Box(  # type: ignore
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float32,
        )

    @property
    def tcp_center(self) -> npt.NDArray[Any]:
        """The COM of the gripper's 2 fingers.

        Returns:
            3-element position.
        """
        right_finger_pos = self.data.body("finger_joint1_tip")
        left_finger_pos = self.data.body("finger_joint1_tip")
        tcp_center = (right_finger_pos.xpos + left_finger_pos.xpos) / 2.0
        return tcp_center

    @property
    def left_pad(self):
        return self.get_body_com("finger_joint1_tip")

    @property
    def right_pad(self):
        return self.get_body_com("finger_joint2_tip")

    def get_endeff_pos(self) -> npt.NDArray[Any]:
        """Returns the position of the end effector."""
        # return self.data.body("hand").xpos
        return self._get_site_pos("endeff")

    def touching_object(self, object_geom_id: int) -> bool:
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id: the ID of the object in question

        Returns:
            Whether the gripper is touching the object
        """

        leftpad_geom_id = self.data.geom("finger1_pad_collision").id
        rightpad_geom_id = self.data.geom("finger2_pad_collision").id

        leftpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in rightpad_object_contacts
        )

        return 0 < leftpad_object_contact_force and 0 < rightpad_object_contact_force

    def _get_pos_goal(self) -> npt.NDArray[Any]:
        """Retrieves goal position from mujoco properties or instance vars.

        Returns:
            Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    @property
    def arm_observation_space(self) -> Box:
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
        if self._partially_observable:
            goal_low = np.zeros(3)
            goal_high = np.zeros(3)
        else:
            assert (
                self.goal_space is not None
            ), "The goal space must be defined to use full observability"
            goal_low = self.goal_space.low
            goal_high = self.goal_space.high
        gripper_low = -1.0
        gripper_high = +1.0
        return Box(
            np.hstack(
                (
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    self._HAND_SPACE.low,
                    gripper_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    self._HAND_SPACE.high,
                    gripper_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

    def parse_gripper_action(
        self, gripper_action: np.float32
    ) -> npt.NDArray[np.float32]:
        """Parses the gripper action.

        Args:
            action (np.float32): The action to parse

        Returns:
            The parsed action
        """
        return np.array([gripper_action, -gripper_action])

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
    ) -> float:
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
        assert (
            self.obj_init_pos is not None
        ), "`obj_init_pos` must be initialized before calling this function."

        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.left_pad
        right_pad = self.right_pad

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            reward_utils.tolerance(
                pad_to_obj_lr[i],  # "x" in the description above
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],  # "margin" in the description above
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid="long_tail",
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        )

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, float(caging_xz))
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + float(reach)) / 2

        return caging_and_gripping

    def set_xyz_action(self, action: npt.NDArray[Any]) -> None:
        """Adjusts the position of the mocap body from the given action.
        Moves each body axis in XYZ by the amount described by the action.

        Args:
            action: The action to apply (in offsets between :math:`[-1, 1]` for each axis in XYZ).
        """
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.mocap_pos = new_mocap_pos
        self.data.mocap_quat = self._HAND_QUAT

    @property
    def gripper_distance_apart(self):
        finger_right, finger_left = (
            self.get_body_com("finger_joint2_tip"),
            self.get_body_com("finger_joint1_tip"),
        )
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
        return gripper_distance_apart

    def touching_object(self, object_geom_id: int) -> bool:
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id: the ID of the object in question

        Returns:
            Whether the gripper is touching the object
        """

        leftpad_geom_id = self.data.geom("finger1_pad_collision").id
        rightpad_geom_id = self.data.geom("finger2_pad_collision").id

        leftpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self.data.efc_force[x.efc_address] for x in rightpad_object_contacts
        )

        return 0 < leftpad_object_contact_force and 0 < rightpad_object_contact_force
