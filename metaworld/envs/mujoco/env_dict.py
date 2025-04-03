"""Dictionaries mapping environment name strings to environment classes,
and organising them into various collections and splits for the benchmarks."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, Literal
from typing import OrderedDict as Typing_OrderedDict
from typing import Sequence, Union

import numpy as np
from typing_extensions import TypeAlias

from metaworld.envs.mujoco.sawyer_xyz import SawyerXYZEnv, v2
from metaworld.envs.mujoco.arm_env import ArmEnv

# Utils

EnvDict: TypeAlias = "Typing_OrderedDict[str, type[ArmEnv]]"
TrainTestEnvDict: TypeAlias = "Typing_OrderedDict[Literal['train', 'test'], EnvDict]"
EnvArgsKwargsDict: TypeAlias = (
    "Dict[str, Dict[Literal['args', 'kwargs'], Union[List, Dict]]]"
)


ALL_V2_ENVIRONMENTS = [
    "assembly-v2",
    "basketball-v2",
    "bin-picking-v2",
    "box-close-v2",
    "button-press-topdown-v2",
    "button-press-topdown-wall-v2",
    "button-press-v2",
    "button-press-wall-v2",
    "coffee-button-v2",
    "coffee-pull-v2",
    "coffee-push-v2",
    "dial-turn-v2",
    "disassemble-v2",
    "door-close-v2",
    "door-lock-v2",
    "door-open-v2",
    "door-unlock-v2",
    "hand-insert-v2",
    "drawer-close-v2",
    "drawer-open-v2",
    "faucet-open-v2",
    "faucet-close-v2",
    "hammer-v2",
    "handle-press-side-v2",
    "handle-press-v2",
    "handle-pull-side-v2",
    "handle-pull-v2",
    "lever-pull-v2",
    "pick-place-wall-v2",
    "pick-out-of-hole-v2",
    "pick-place-v2",
    "plate-slide-v2",
    "plate-slide-side-v2",
    "plate-slide-back-v2",
    "plate-slide-back-side-v2",
    "peg-insert-side-v2",
    "peg-unplug-side-v2",
    "soccer-v2",
    "stick-push-v2",
    "stick-pull-v2",
    "push-v2",
    "push-wall-v2",
    "push-back-v2",
    "reach-v2",
    "reach-wall-v2",
    "shelf-place-v2",
    "sweep-into-v2",
    "sweep-v2",
    "window-open-v2",
    "window-close-v2",
]

SAWYER_ENVIRONMENTS = OrderedDict(
    (
        ("assembly-v2", v2.SawyerNutAssemblyEnvV2),
        ("basketball-v2", v2.SawyerBasketballEnvV2),
        ("bin-picking-v2", v2.SawyerBinPickingEnvV2),
        ("box-close-v2", v2.SawyerBoxCloseEnvV2),
        ("button-press-topdown-v2", v2.SawyerButtonPressTopdownEnvV2),
        ("button-press-topdown-wall-v2", v2.SawyerButtonPressTopdownWallEnvV2),
        ("button-press-v2", v2.SawyerButtonPressEnvV2),
        ("button-press-wall-v2", v2.SawyerButtonPressWallEnvV2),
        ("coffee-button-v2", v2.SawyerCoffeeButtonEnvV2),
        ("coffee-pull-v2", v2.SawyerCoffeePullEnvV2),
        ("coffee-push-v2", v2.SawyerCoffeePushEnvV2),
        ("dial-turn-v2", v2.SawyerDialTurnEnvV2),
        ("disassemble-v2", v2.SawyerNutDisassembleEnvV2),
        ("door-close-v2", v2.SawyerDoorCloseEnvV2),
        ("door-lock-v2", v2.SawyerDoorLockEnvV2),
        ("door-open-v2", v2.SawyerDoorEnvV2),
        ("door-unlock-v2", v2.SawyerDoorUnlockEnvV2),
        ("hand-insert-v2", v2.SawyerHandInsertEnvV2),
        ("drawer-close-v2", v2.SawyerDrawerCloseEnvV2),
        ("drawer-open-v2", v2.SawyerDrawerOpenEnvV2),
        ("faucet-open-v2", v2.SawyerFaucetOpenEnvV2),
        ("faucet-close-v2", v2.SawyerFaucetCloseEnvV2),
        ("hammer-v2", v2.SawyerHammerEnvV2),
        ("handle-press-side-v2", v2.SawyerHandlePressSideEnvV2),
        ("handle-press-v2", v2.SawyerHandlePressEnvV2),
        ("handle-pull-side-v2", v2.SawyerHandlePullSideEnvV2),
        ("handle-pull-v2", v2.SawyerHandlePullEnvV2),
        ("lever-pull-v2", v2.SawyerLeverPullEnvV2),
        ("peg-insert-side-v2", v2.SawyerPegInsertionSideEnvV2),
        ("pick-place-wall-v2", v2.SawyerPickPlaceWallEnvV2),
        ("pick-out-of-hole-v2", v2.SawyerPickOutOfHoleEnvV2),
        ("reach-v2", v2.SawyerReachEnvV2),
        ("push-back-v2", v2.SawyerPushBackEnvV2),
        ("push-v2", v2.SawyerPushEnvV2),
        ("pick-place-v2", v2.SawyerPickPlaceEnvV2),
        ("plate-slide-v2", v2.SawyerPlateSlideEnvV2),
        ("plate-slide-side-v2", v2.SawyerPlateSlideSideEnvV2),
        ("plate-slide-back-v2", v2.SawyerPlateSlideBackEnvV2),
        ("plate-slide-back-side-v2", v2.SawyerPlateSlideBackSideEnvV2),
        ("peg-unplug-side-v2", v2.SawyerPegUnplugSideEnvV2),
        ("soccer-v2", v2.SawyerSoccerEnvV2),
        ("stick-push-v2", v2.SawyerStickPushEnvV2),
        ("stick-pull-v2", v2.SawyerStickPullEnvV2),
        ("push-wall-v2", v2.SawyerPushWallEnvV2),
        ("reach-wall-v2", v2.SawyerReachWallEnvV2),
        ("shelf-place-v2", v2.SawyerShelfPlaceEnvV2),
        ("sweep-into-v2", v2.SawyerSweepIntoGoalEnvV2),
        ("sweep-v2", v2.SawyerSweepEnvV2),
        ("window-open-v2", v2.SawyerWindowOpenEnvV2),
        ("window-close-v2", v2.SawyerWindowCloseEnvV2),
    )
)

# ... Other envs


ARM_NAMES = ["sawyer"]

ARM_ENV_CLS_MAPS = [SAWYER_ENVIRONMENTS]

ARM_INDICES = OrderedDict([(arm, index) for index, arm in enumerate(ARM_NAMES)])


def _get_env_dict(env_names: Sequence[str], arm_name: str = "sawyer") -> EnvDict:
    """Returns an `OrderedDict` containing `(env_name, env_cls)` tuples for the given env_names and arm.

    Args:
        env_names: The environment names
        arm: The arm to use for the environments

    Returns:
        The appropriate `OrderedDict.
    """
    ENV_CLS_MAP = ARM_ENV_CLS_MAPS[ARM_INDICES[arm_name]]
    return OrderedDict(
        [
            (env_name, env_cls)
            for env_name, env_cls in ENV_CLS_MAP.items()
            if env_name in env_names
        ]
    )


def _get_train_test_env_dict(
    train_env_names: Sequence[str],
    test_env_names: Sequence[str],
    arm_name: str = "sawyer",
) -> TrainTestEnvDict:
    """Returns an `OrderedDict` containing two sub-keys ("train" and "test" at positions 0 and 1),
    each containing the appropriate `OrderedDict` for the train and test classes of the benchmark.

    Args:
        train_env_names: The train environment names.
        test_env_names: The test environment names
        arm: The arm to use for the environments

    Returns:
        The appropriate `OrderedDict`.
    """
    return OrderedDict(
        (
            ("train", _get_env_dict(train_env_names, arm_name)),
            ("test", _get_env_dict(test_env_names, arm_name)),
        )
    )


def _get_args_kwargs(all_envs: Sequence[str], env_subset: EnvDict) -> EnvArgsKwargsDict:
    """Returns containing a `dict` of "args" and "kwargs" for each environment in a given list of environments.
    Specifically, sets an empty "args" array and a "kwargs" dictionary with a "task_id" key for each env.

    Args:
        all_envs: The full list of envs
        env_subset: The subset of envs to get args and kwargs for

    Returns:
        The args and kwargs dictionary.
    """
    return {
        key: dict(args=[], kwargs={"task_id": list(all_envs).index(key)})
        for key, _ in env_subset.items()
    }


# MT Dicts

MT10_V2_ENVIRONMENTS = [
    "reach-v2",
    "push-v2",
    "pick-place-v2",
    "door-open-v2",
    "drawer-open-v2",
    "drawer-close-v2",
    "button-press-topdown-v2",
    "peg-insert-side-v2",
    "window-open-v2",
    "window-close-v2",
]
MT10_V2 = OrderedDict(
    [
        (arm_name, _get_env_dict(MT10_V2_ENVIRONMENTS, arm_name))
        for arm_name in ARM_NAMES
    ]
)
MT10_V2_ARGS_KWARGS = OrderedDict(
    [
        (arm_name, _get_args_kwargs(MT10_V2[arm_name], MT10_V2[arm_name]))
        for arm_name in ARM_NAMES
    ]
)

# MT50_V2 = ALL_V2_ENVIRONMENTS
# MT50_V2_ARGS_KWARGS = _get_args_kwargs(ALL_V2_ENVIRONMENTS, MT50_V2)
MT50_V2 = OrderedDict(
    [(arm_name, _get_env_dict(ALL_V2_ENVIRONMENTS, arm_name)) for arm_name in ARM_NAMES]
)
MT50_V2_ARGS_KWARGS = OrderedDict(
    [
        (arm_name, _get_args_kwargs(MT50_V2[arm_name], MT50_V2[arm_name]))
        for arm_name in ARM_NAMES
    ]
)

# ML Dicts

ML1_V2 = OrderedDict(
    [(arm_name, _get_env_dict(ALL_V2_ENVIRONMENTS, arm_name)) for arm_name in ARM_NAMES]
)

ML10_V2 = OrderedDict(
    [
        (
            arm_name,
            _get_train_test_env_dict(
                train_env_names=[
                    "reach-v2",
                    "push-v2",
                    "pick-place-v2",
                    "door-open-v2",
                    "drawer-close-v2",
                    "button-press-topdown-v2",
                    "peg-insert-side-v2",
                    "window-open-v2",
                    "sweep-v2",
                    "basketball-v2",
                ],
                test_env_names=[
                    "drawer-open-v2",
                    "door-close-v2",
                    "shelf-place-v2",
                    "sweep-into-v2",
                    "lever-pull-v2",
                ],
                arm_name=arm_name,
            ),
        )
        for arm_name in ARM_NAMES
    ]
)
ML10_ARGS_KWARGS = OrderedDict(
    [
        (
            arm_name,
            {
                "train": _get_args_kwargs(
                    ALL_V2_ENVIRONMENTS, ML10_V2[arm_name]["train"]
                ),
                "test": _get_args_kwargs(
                    ALL_V2_ENVIRONMENTS, ML10_V2[arm_name]["test"]
                ),
            },
        )
        for arm_name in ARM_NAMES
    ]
)

ML45_V2 = OrderedDict(
    [
        (
            arm_name,
            _get_train_test_env_dict(
                train_env_names=[
                    "assembly-v2",
                    "basketball-v2",
                    "button-press-topdown-v2",
                    "button-press-topdown-wall-v2",
                    "button-press-v2",
                    "button-press-wall-v2",
                    "coffee-button-v2",
                    "coffee-pull-v2",
                    "coffee-push-v2",
                    "dial-turn-v2",
                    "disassemble-v2",
                    "door-close-v2",
                    "door-open-v2",
                    "drawer-close-v2",
                    "drawer-open-v2",
                    "faucet-open-v2",
                    "faucet-close-v2",
                    "hammer-v2",
                    "handle-press-side-v2",
                    "handle-press-v2",
                    "handle-pull-side-v2",
                    "handle-pull-v2",
                    "lever-pull-v2",
                    "pick-place-wall-v2",
                    "pick-out-of-hole-v2",
                    "push-back-v2",
                    "pick-place-v2",
                    "plate-slide-v2",
                    "plate-slide-side-v2",
                    "plate-slide-back-v2",
                    "plate-slide-back-side-v2",
                    "peg-insert-side-v2",
                    "peg-unplug-side-v2",
                    "soccer-v2",
                    "stick-push-v2",
                    "stick-pull-v2",
                    "push-wall-v2",
                    "push-v2",
                    "reach-wall-v2",
                    "reach-v2",
                    "shelf-place-v2",
                    "sweep-into-v2",
                    "sweep-v2",
                    "window-open-v2",
                    "window-close-v2",
                ],
                test_env_names=[
                    "bin-picking-v2",
                    "box-close-v2",
                    "hand-insert-v2",
                    "door-lock-v2",
                    "door-unlock-v2",
                ],
                arm_name=arm_name,
            ),
        )
        for arm_name in ARM_NAMES
    ]
)
ML45_ARGS_KWARGS = OrderedDict(
    [
        (
            arm_name,
            {
                "train": _get_args_kwargs(
                    ALL_V2_ENVIRONMENTS, ML45_V2[arm_name]["train"]
                ),
                "test": _get_args_kwargs(
                    ALL_V2_ENVIRONMENTS, ML45_V2[arm_name]["test"]
                ),
            },
        )
        for arm_name in ARM_NAMES
    ]
)
