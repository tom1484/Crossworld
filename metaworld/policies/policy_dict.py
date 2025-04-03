from __future__ import annotations

import re
from collections import OrderedDict

from metaworld.policies.policy import Policy
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS, ARM_NAMES


def import_policy(arm_name: str, env_name: str) -> Policy | None:
    arm_name = re.sub(r"-", r"_", arm_name)
    env_name = re.sub(r"-", r"_", env_name)
    module_name = f"metaworld.policies.{arm_name}.{arm_name}_{env_name}_policy"
    # import the module dynamically
    try:
        module = __import__(module_name, fromlist=[""])
        return module
    except ImportError as e:
        return None


ARM_POLICY_CLS_MAPS = [
    OrderedDict(
        [
            (env_name, import_policy(arm_name, env_name))
            for env_name in ALL_V2_ENVIRONMENTS
        ]
    )
    for arm_name in ARM_NAMES
]


__all__ = [
    "ARM_POLICY_CLS_MAPS",
]
