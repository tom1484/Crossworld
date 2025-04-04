from __future__ import annotations

import re
from collections import OrderedDict

from metaworld.policies.policy import Policy
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS, ARM_NAMES


def import_policy(arm_name: str, env_name: str) -> Policy | None:
    arm_name = re.sub(r"-", r"_", arm_name)
    env_name = re.sub(r"-", r"_", env_name)

    module_name = f"{arm_name}_{env_name}_policy"
    cls_name = "".join(word.capitalize() for word in module_name.split("_"))

    # import the module dynamically
    module_path = f"metaworld.policies.{arm_name}.{arm_name}_{env_name}_policy"
    try:
        module = __import__(module_path, fromlist=[""])
        cls = getattr(module, cls_name)
        if issubclass(cls, Policy):
            return cls
        else:
            raise ImportError(f"{cls_name} is not a subclass of Policy")
    except ImportError as e:
        return None


SAWYER_POLICIES = OrderedDict(
    [(env_name, import_policy("sawyer", env_name)) for env_name in ALL_V2_ENVIRONMENTS]
)

PANDA_POLICIES = OrderedDict(
    [(env_name, import_policy("panda", env_name)) for env_name in ALL_V2_ENVIRONMENTS]
)

ARM_POLICY_CLS_MAPS = OrderedDict(
    (("sawyer", SAWYER_POLICIES), ("panda", PANDA_POLICIES))
)


__all__ = [
    "ARM_POLICY_CLS_MAPS",
]
