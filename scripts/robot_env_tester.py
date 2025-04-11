#!/usr/bin/env python3
"""
Universal environment and policy tester for CrossWorld.
Supports testing environments across different robot arms (Sawyer, Panda, Jaco)
and provides a framework for testing and developing scripted policies.

This version is designed for headless servers without monitors, using
PyGame-based keyboard control and saving rendered frames as images.

Usage:
    python robot_env_tester.py --robot-type sawyer --env-name door-open-v2 --policy-test
    python robot_env_tester.py --robot-type panda --env-name drawer-close-v2 --pygame-control
    python robot_env_tester.py --robot-type jaco --env-name button-press-v2 --list-envs
    python robot_env_tester.py --robot-type sawyer --env-name pick-place-v2 --task 3
    python robot_env_tester.py --robot-type sawyer --env-name pick-place-v2 --pygame-control
"""

import argparse
import importlib
import time
import random
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar
import numpy as np
from collections import OrderedDict
from datetime import datetime

import crossworld
from crossworld import Task
from crossworld.policies.policy import Policy
from crossworld.policies.policy_dict import ARM_POLICY_CLS_MAPS

# Add PyGame import for interactive control
try:
    import pygame
    from pygame.locals import (
        KEYDOWN,
        QUIT,
        K_w,
        K_a,
        K_s,
        K_d,
        K_q,
        K_e,
        K_z,
        K_c,
        K_k,
        K_j,
        K_h,
        K_l,
        K_x,
        K_r,
        K_p,
    )

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Remove the problematic MjViewer import

# Dictionary mapping environment names to their policy classes (when available)
# POLICY_MAP: Dict[str, str] = {
#     # Sawyer policies
#     "door-open-v2": "SawyerDoorOpenV2Policy",
#     "door-close-v2": "SawyerDoorCloseV2Policy",
#     "door-lock-v2": "SawyerDoorLockV2Policy",
#     "door-unlock-v2": "SawyerDoorUnlockV2Policy",
#     "drawer-open-v2": "SawyerDrawerOpenV2Policy",
#     "drawer-close-v2": "SawyerDrawerCloseV2Policy",
#     "button-press-v2": "SawyerButtonPressV2Policy",
#     "peg-insert-side-v2": "SawyerPegInsertSideV2Policy",
#     "window-open-v2": "SawyerWindowOpenV2Policy",
#     "window-close-v2": "SawyerWindowCloseV2Policy",
#     # Add more mappings as needed
# }

# Robot-specific environment prefix mapping
ROBOT_ENV_PREFIX: Dict[str, str] = {
    "sawyer": "",  # Sawyer is the default, no prefix needed
    "panda": "panda_",
    "jaco": "jaco_",
}

# Type definitions
EnvType = TypeVar("EnvType")
PolicyType = TypeVar("PolicyType")
ObsType = np.ndarray
ActionType = np.ndarray
BenchmarkType = Any  # crossworld.Benchmark


def get_policy(env_name: str, arm_name: str) -> Optional[Any]:
    """Attempt to load a policy for the given environment name."""
    POLICY_CLS_MAP = ARM_POLICY_CLS_MAPS.get(arm_name, {})
    policy_class: Policy = POLICY_CLS_MAP.get(env_name)
    if policy_class is None:
        return None

    return policy_class()


def get_environment_names(benchmark_class: BenchmarkType) -> List[str]:
    """Get all available environment names from a benchmark."""
    envs: List[str] = []
    for env_name in benchmark_class.train_classes.keys():
        envs.append(env_name)
    return sorted(envs)


def get_tasks_for_env(benchmark: BenchmarkType, env_name: str) -> List[Task]:
    """Get all tasks for a specific environment.

    Args:
        benchmark: The benchmark containing tasks
        env_name: The environment name to filter tasks for

    Returns:
        List of matching tasks
    """
    matching_tasks: List[Task] = [
        t for t in benchmark.train_tasks if t.env_name == env_name
    ]
    return matching_tasks


def list_tasks_for_env(benchmark: BenchmarkType, env_name: str) -> None:
    """Print information about available tasks for an environment.

    Args:
        benchmark: The benchmark containing tasks
        env_name: The environment name to show tasks for
    """
    matching_tasks = get_tasks_for_env(benchmark, env_name)
    if not matching_tasks:
        print(f"No tasks found for environment: {env_name}")
        return

    print(f"\nFound {len(matching_tasks)} tasks for environment: {env_name}")
    print(f"Use --task INDEX to select a specific task (0-{len(matching_tasks)-1})")

    # Display a few example tasks
    if len(matching_tasks) > 3:
        print("\nFirst 3 tasks:")
        for i in range(3):
            print(f"  Task {i}: {matching_tasks[i].env_name}")
        print(f"  ... and {len(matching_tasks)-3} more")
    else:
        print("\nAvailable tasks:")
        for i, task in enumerate(matching_tasks):
            print(f"  Task {i}: {task.env_name}")


def save_render_frames(
    output_dir: str,
    frames: List[np.ndarray],
    prefix: str = "frame",
    save_raw: bool = False,
):
    """Save rendered frames as images and combine them into a video.

    Args:
        output_dir: Directory to save images and video
        frames: List of frames to save and combine
        prefix: Filename prefix
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    raw_dir: str = None
    if save_raw:
        raw_dir = os.path.join(output_dir, "raw")
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

    # Save individual frames as images
    if raw_dir:
        for i, frame in enumerate(frames):
            filename: str = f"{prefix}_{i:04d}.png"
            filepath: str = os.path.join(raw_dir, filename)

            # Save the image using PIL
            try:
                from PIL import Image

                Image.fromarray(frame).save(filepath)
            except ImportError:
                print("PIL not available, trying cv2...")
                try:
                    import cv2

                    # OpenCV expects BGR format
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        bgr_array = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(filepath, bgr_array)
                    else:
                        cv2.imwrite(filepath, frame)
                except ImportError:
                    print(
                        "WARNING: Neither PIL nor OpenCV available. Cannot save rendered frames."
                    )

    # Combine all frames into a GIF
    try:
        from PIL import Image

        if not frames:
            print("No frames to combine into GIF.")
            return

        gif_filepath = os.path.join(output_dir, "episode.gif")

        # Convert each frame (assumed to be in RGB format) into a PIL Image
        pil_frames = []
        for frame in frames:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                pil_frame = Image.fromarray(frame)
            else:
                # For non-RGB frames, simply convert without mode change
                pil_frame = Image.fromarray(frame)
            pil_frames.append(pil_frame)

        # Save as an animated GIF, duration is set to 100ms per frame
        pil_frames[0].save(
            gif_filepath,
            save_all=True,
            append_images=pil_frames[1:],
            loop=0,
            duration=100,
        )
        print(f"GIF saved to {gif_filepath}")
    except ImportError:
        print("PIL not available, skipping GIF creation.")


def render_frame(env: EnvType, frames: List[np.ndarray]) -> None:
    """
    Render the environment and append the frame in a list.

    Args:
        env: The environment to render
        frames: List to store rendered frames
    """
    # Try different rendering approaches to handle API differences
    rgb_array: Optional[np.ndarray] = None

    # For CrossWorld environments, first try without any arguments
    # This should work with initialized render_mode
    try:
        rgb_array = env.render()
    except (TypeError, ValueError, AttributeError):
        if rgb_array is None:
            print(
                "Warning: Could not render the environment - all rendering methods failed"
            )
            return

    frames.append(rgb_array)


# Add common PyGame setup functions
def init_pygame_common(window_size: Tuple[int, int]) -> Tuple[Any, Any]:
    import pygame

    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("CrossWorld Progress")
    clock = pygame.time.Clock()
    return screen, clock


def process_common_pygame_events(clock: Any) -> bool:
    import pygame

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_SPACE:
                print("Paused. Press SPACE to resume or ESC to quit.")
                paused = True
                while paused:
                    for evt in pygame.event.get():
                        if evt.type == pygame.KEYDOWN:
                            if evt.key == pygame.K_SPACE:
                                paused = False
                            elif evt.key == pygame.K_ESCAPE:
                                return False
                    clock.tick(10)
    return True


def record_trajectory(
    env: EnvType,
    policy: Optional[PolicyType] = None,
    max_steps: Optional[int] = None,
    loop: bool = False,
    output_dir: Optional[str] = None,
    delay: float = 0.0,
    window_size: Tuple[int, int] = (640, 480),
) -> Dict[str, Any]:
    """
    Execute an environment trajectory using the provided policy or random actions.

    Args:
        env: The environment to run
        policy: Optional policy to generate actions
        max_steps: Maximum number of steps to run
        output_dir: Directory to save rendered frames (None = no saving)
        delay: Time delay between steps

    Returns:
        Dictionary containing trajectory data
    """
    # FIX: I really don't know why this is needed
    import numpy as np

    obs: np.ndarray
    info: Dict
    done: bool
    count: int
    success: bool

    def reset():
        nonlocal obs, info, done, count, success

        obs, info = env.reset()
        done = False
        count = 0
        success = False

    reset()

    # Save initial render
    rendered_frames: List[np.ndarray] = []
    render_frame(env, rendered_frames)

    trajectory_data: Dict[str, Any] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "next_observations": [],
        "success": False,
    }

    # Initialize minimal PyGame display if available
    if PYGAME_AVAILABLE:
        screen, clock = init_pygame_common(window_size)

    if loop:
        max_steps = None
    
    while (max_steps is None or count < max_steps) and not done:
        if policy:
            action = policy.get_action(obs)
        else:
            # zero action
            action = np.zeros(env.action_space.shape[0], dtype=np.float32)

        trajectory_data["observations"].append(obs)
        trajectory_data["actions"].append(action)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trajectory_data["rewards"].append(reward)
        trajectory_data["next_observations"].append(next_obs)

        # If output_dir provided, save frame (using existing render_frame if applicable)
        if output_dir:
            render_frame(env, rendered_frames)

        # --- New PyGame integration for progress display ---
        if PYGAME_AVAILABLE:
            if not process_common_pygame_events(clock):
                print("Terminated by user via PyGame event.")
                break
            try:
                rgb_array = env.render()
                if rgb_array is not None:
                    # Convert image for PyGame display
                    import pygame
                    import numpy as np

                    arr = (
                        np.transpose(rgb_array, (1, 0, 2))
                        if len(rgb_array.shape) == 3
                        else rgb_array
                    )
                    surface = pygame.surfarray.make_surface(arr)
                    surface = pygame.transform.scale(surface, window_size)
                    screen.blit(surface, (0, 0))
                    pygame.display.update()
                    clock.tick(60)
            except Exception as e:
                print(f"Warning: Could not update PyGame display: {e}")
        # --- End PyGame integration ---

        print(f"Step: {count}, Reward: {reward:.4f}")
        print(f"Action: {action}")
        if "success" in info:
            print(f"Success: {info['success']}")
            if info["success"] and not success:
                success = True
                print("Success achieved!")

        obs = next_obs
        if delay > 0:
            time.sleep(delay)
        count += 1

        if done and loop:
            reset()

    trajectory_data["success"] = success

    if PYGAME_AVAILABLE:
        import pygame

        pygame.quit()

    # Save rendered frames into a video if output_dir is provided
    if not loop and output_dir and rendered_frames:
        save_render_frames(output_dir, rendered_frames, prefix="frame")
        print(f"Rendered frames saved to {output_dir}")

    return trajectory_data


def keyboard_control(
    env: EnvType,
    max_steps: int = 1000,
    output_dir: Optional[str] = None,
    step_delay: float = 0.01,
    window_size: Tuple[int, int] = (640, 480),
) -> None:
    """
    Provide PyGame-based interactive keyboard control of the environment.

    Args:
        env: The environment to control
        max_steps: Maximum steps to run
        output_dir: Directory to save rendered frames
        step_delay: Time delay between steps (seconds)
        window_size: PyGame window size (width, height)
    """
    if not PYGAME_AVAILABLE:
        print(
            "Error: PyGame is not available. Please install pygame with 'pip install pygame'."
        )
        return

    # Use common PyGame initialization
    screen, clock = init_pygame_common(window_size)

    # Define key to action mapping
    char_to_action = {
        K_w: np.array([0, -1, 0, 0]),  # Forward
        K_a: np.array([1, 0, 0, 0]),  # Left
        K_s: np.array([0, 1, 0, 0]),  # Backward
        K_d: np.array([-1, 0, 0, 0]),  # Right
        K_q: np.array([1, -1, 0, 0]),  # Forward-left
        K_e: np.array([-1, -1, 0, 0]),  # Forward-right
        K_z: np.array([1, 1, 0, 0]),  # Backward-left
        K_c: np.array([-1, 1, 0, 0]),  # Backward-right
        K_k: np.array([0, 0, 1, 0]),  # Up
        K_j: np.array([0, 0, -1, 0]),  # Down
    }

    # Dictionary to map key codes to readable names
    key_names = {
        K_w: "W (Forward)",
        K_a: "A (Left)",
        K_s: "S (Backward)",
        K_d: "D (Right)",
        K_q: "Q (Forward-Left)",
        K_e: "E (Forward-Right)",
        K_z: "Z (Back-Left)",
        K_c: "C (Back-Right)",
        K_k: "K (Up)",
        K_j: "J (Down)",
        K_h: "H (Close Gripper)",
        K_l: "L (Open Gripper)",
        K_x: "X (Toggle Lock)",
        K_r: "R (Reset)",
        pygame.K_SPACE: "SPACE (Record Keypoint)",
        pygame.K_ESCAPE: "ESC (Quit)",
    }

    # Initialize action with zeros
    action_dim: int = env.action_space.shape[0]
    action: np.ndarray = np.zeros(action_dim)

    # Get action space bounds
    action_low: np.ndarray = env.action_space.low
    action_high: np.ndarray = env.action_space.high

    # Variable to track the last pressed key
    last_key_pressed = None
    last_key_time = 0  # When the key was pressed (for fading effect)

    print("\nPyGame Keyboard Control Mode")
    print(f"Action space dimension: {action_dim}")
    print("\nControls:")
    print("- WASD: Move in XY plane")
    print("- Q/E/Z/C: Diagonal movement in XY plane")
    print("- K/J: Move up/down (Z axis)")
    print("- H: Close gripper")
    print("- L: Open gripper")
    print("- R: Reset environment")
    print("- X: Toggle action lock")
    print("- SPACE: Record current position as keypoint")  # Added spacebar instruction
    print("- ESC: Quit")

    obs: np.ndarray = env.reset()
    step_count: int = 0
    clock = pygame.time.Clock()

    # Save initial render
    rendered_frames: List[np.ndarray] = []
    render_frame(env, rendered_frames)

    # Lists to store recorded keypoints instead of every step
    keypoint_observations: List[np.ndarray] = []
    keypoint_actions: List[np.ndarray] = []
    keypoint_positions: List[np.ndarray] = []  # Store end effector positions

    # Initialize control variables
    lock_action = False
    running = True
    recording_flash_time = 0

    # while running and step_count < max_steps:
    while True:
        # Process PyGame events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                break
            elif event.type == KEYDOWN:
                # Record which key was pressed
                last_key_pressed = event.key
                last_key_time = 30  # Display for 30 frames

                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                elif event.key == K_r:
                    print("Resetting environment")
                    obs = env.reset()
                    step_count = 0
                    action = np.zeros(action_dim)
                elif event.key == K_x:
                    lock_action = not lock_action
                    print(f"Action lock: {'ON' if lock_action else 'OFF'}")
                elif event.key == K_h:
                    # Close gripper
                    if action_dim > 3:
                        action[3] = 1
                elif event.key == K_l:
                    # Open gripper
                    if action_dim > 3:
                        action[3] = -1
                elif event.key == pygame.K_SPACE:
                    # Record current keypoint when spacebar is pressed
                    keypoint_observations.append(obs.copy())
                    keypoint_actions.append(action.copy())

                    # Extract end effector position if available
                    ee_pos = None
                    if hasattr(env, "get_endeff_pos"):
                        ee_pos = env.get_endeff_pos()
                    elif "hand_pos" in obs:
                        ee_pos = obs["hand_pos"]
                    elif len(obs) >= 3:  # Assume first 3 elements might be position
                        ee_pos = obs[:3]

                    if ee_pos is not None:
                        keypoint_positions.append(ee_pos)

                    print(f"Keypoint recorded: {len(keypoint_observations)}")
                    print(f"End effector position: {ee_pos}")

                    # Set flash time for visual feedback
                    recording_flash_time = 30  # Show visual feedback for 30 frames

                elif event.key in char_to_action:
                    # Apply movement based on key
                    key_action = char_to_action[event.key]
                    # Only apply the action dimensions that fit
                    valid_dims = min(len(key_action), action_dim)
                    if not lock_action:
                        # Reset XYZ movement unless locked
                        action[:3] = 0
                    # Apply the movement action
                    action[:valid_dims] += (
                        key_action[:valid_dims] * 0.1
                    )  # Scale factor for sensitivity

        # Check if we should continue
        if not running:
            break

        # Clip action to valid range
        action = np.clip(action, action_low, action_high)

        # Apply the action to the environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Update PyGame window title with current info
        success_info = f"SUCCESS!" if info.get("success", False) else ""
        pygame.display.set_caption(
            f"Step: {step_count} | Keypoints: {len(keypoint_observations)} | Reward: {reward:.4f} | {success_info}"
        )

        # Try to show the rendered image in the PyGame window
        try:
            # Get the rendered image
            rgb_array = env.render()
            if rgb_array is not None:
                # Convert the numpy array to a PyGame surface
                rgb_array = (
                    np.transpose(rgb_array, (1, 0, 2))
                    if len(rgb_array.shape) == 3
                    else rgb_array
                )
                surface = pygame.surfarray.make_surface(rgb_array)
                # Scale the image to fit the window
                surface = pygame.transform.scale(surface, window_size)

                # Draw recording flash if active
                if recording_flash_time > 0:
                    # Create a semi-transparent green overlay for visual feedback
                    flash_surface = pygame.Surface(window_size, pygame.SRCALPHA)
                    flash_surface.fill(
                        (0, 255, 0, min(recording_flash_time * 3, 100))
                    )  # Green with fading alpha
                    surface.blit(flash_surface, (0, 0))
                    recording_flash_time -= 1

                # Display the image
                screen.blit(surface, (0, 0))

                # Create a font object
                font = pygame.font.Font(None, 36)

                # Draw keypoint count
                text = font.render(
                    f"Keypoints: {len(keypoint_observations)}", True, (255, 255, 255)
                )
                text_bg = pygame.Surface(
                    (text.get_width() + 20, text.get_height() + 10), pygame.SRCALPHA
                )
                text_bg.fill((0, 0, 0, 128))
                screen.blit(text_bg, (10, 10))
                screen.blit(text, (20, 15))

                # Draw last pressed key if available
                if last_key_time > 0:
                    key_text = f"Key: {key_names.get(last_key_pressed, 'Unknown')}"
                    key_label = font.render(key_text, True, (255, 255, 0))
                    key_bg = pygame.Surface(
                        (key_label.get_width() + 20, key_label.get_height() + 10),
                        pygame.SRCALPHA,
                    )
                    key_bg.fill((0, 0, 0, min(128, last_key_time * 4)))  # Fade out
                    screen.blit(key_bg, (10, 60))
                    screen.blit(key_label, (20, 65))
                    last_key_time -= 1

                pygame.display.flip()
        except Exception as e:
            print(f"Warning: Could not render to PyGame window: {e}")

        # Save render frame if requested and a keypoint was just recorded
        if output_dir and recording_flash_time == 29:  # Just after recording
            render_frame(env, rendered_frames)

        # Display text information
        print(f"Step: {step_count}")
        print(f"Action taken: {action}")
        print(f"Reward: {reward:.4f}")
        if "success" in info:
            print(f"Success: {info['success']}")

        # Update state for next iteration
        obs = next_obs
        step_count += 1

        if terminated or truncated:
            print("Episode finished")
            obs = env.reset()

        # Control the frame rate
        clock.tick(60)  # 60 FPS
        time.sleep(step_delay)  # Add additional delay if needed

    # Save rendered frames into a video if output_dir is provided
    if output_dir and rendered_frames:
        save_render_frames(output_dir, rendered_frames, prefix="frame")
        print(f"Rendered frames saved to {output_dir}")

    # Save recorded keypoints if we have any
    if (
        keypoint_observations
        and input("Save recorded keypoints? (y/n): ").lower() == "y"
    ):
        arm_name: str = input("Robot type (sawyer/panda/jaco): ").strip().lower()
        env_name: str = input("Environment name: ").strip()
        filename: str = f"{arm_name}_{env_name.replace('-', '_')}_keypoints.npy"

        save_dir: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "policy_recordings"
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_data: Dict[str, Any] = {
            "observations": np.array(keypoint_observations),
            "actions": np.array(keypoint_actions),
            "positions": np.array(keypoint_positions) if keypoint_positions else None,
            "arm_name": arm_name,
            "env_name": env_name,
            "recorded_steps": len(keypoint_observations),
        }

        save_path: str = os.path.join(save_dir, filename)
        np.save(save_path, save_data)
        print(f"Saved {len(keypoint_observations)} keypoints to: {save_path}")

    # Clean up PyGame
    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test CrossWorld environments and policies"
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="sawyer",
        choices=["sawyer", "panda", "jaco"],
        help="Robot arm name (sawyer, panda, jaco)",
    )
    parser.add_argument(
        "--env-name", type=str, help="Environment name (e.g., door-open-v2)"
    )
    parser.add_argument(
        "--task", type=int, default=0, help="Task index to use (default: 0)"
    )
    parser.add_argument(
        "--list-envs", action="store_true", help="List available environments"
    )
    parser.add_argument(
        "--render-test",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--policy-test",
        action="store_true",
        help="Test with appropriate scripted policy if available",
    )
    parser.add_argument(
        "--pygame-control",
        action="store_true",
        help="Enable PyGame-based keyboard control interface",
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Maximum number of steps to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save rendered frames (default: timestamped directory)",
    )
    parser.add_argument(
        "--include-test-tasks",
        action="store_true",
        default=False,
        help="Include test tasks when loading benchmarks (slower, may cause XML errors)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ml10",
        choices=["mt1", "mt10", "mt50", "ml1", "ml10", "ml45"],
        help="Benchmark to use (default: ml10 for faster loading)",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default="640,480",
        help="PyGame window size (width,height)",
    )
    parser.add_argument(
        "--camera-name",
        type=str,
        default="frontview",
        choices=["corner", "corner2", "corner3", "frontview", "leftview", "rightview", "topview"],
        help="Name of camera for rendering (default: frontview)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay between steps in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save rendered frames as images and combine into a video",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the environment indefinitely until ESC is pressed",
    )

    args = parser.parse_args()

    # Parse window size
    try:
        width, height = map(int, args.window_size.split(","))
        window_size = (width, height)
    except:
        print("Warning: Invalid window size format. Using default 640x480.")
        window_size = (640, 480)

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set up output directory if rendering is enabled
    output_dir: Optional[str] = None
    if args.save:
        output_dir = args.output_dir
        if output_dir is None:
            timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./render_output/{args.arm_name}_{args.env_name}/{timestamp}"

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except Exception as e:
                print(f"Warning: Could not create output directory: {e}")
                output_dir = None

    # Initialize benchmark based on user's choice
    benchmark: Optional[BenchmarkType] = None
    benchmark_initializers: Dict[str, Callable[[], BenchmarkType]] = {
        "mt1": lambda: crossworld.MT1(
            env_name=args.env_name,
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "mt10": lambda: crossworld.MT10(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "mt50": lambda: crossworld.MT50(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "ml1": lambda: crossworld.ML1(
            env_name=args.env_name,
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "ml10": lambda: crossworld.ML10(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "ml45": lambda: crossworld.ML45(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
    }

    # Try the user's selected benchmark first
    try:
        print(f"Initializing {args.benchmark.upper()} benchmark...")
        benchmark = benchmark_initializers[args.benchmark]()
        print(f"Successfully loaded {args.benchmark.upper()} benchmark")
    except Exception as error:
        print("Error: Could not initialize benchmark:", error)
        sys.exit(1)

    # List environments if requested
    if args.list_envs:
        env_names: List[str] = get_environment_names(benchmark)
        print("\nAvailable environments:")

        POLICY_CLS_MAP = ARM_POLICY_CLS_MAPS.get(args.arm_name, {})
        for env_name in env_names:
            policy_available: bool = env_name in POLICY_CLS_MAP
            arm_name: str = "sawyer"  # Default
            if env_name.startswith("panda_"):
                arm_name = "panda"
                base_name: str = env_name[6:]  # Remove panda_ prefix
            elif env_name.startswith("jaco_"):
                arm_name = "jaco"
                base_name = env_name[5:]  # Remove jaco_ prefix
            else:
                base_name = env_name

            print(
                f"  {env_name} (robot: {arm_name}) {'(policy available)' if policy_available else ''}"
            )
        return

    # Validate environment name
    if not args.env_name:
        print(
            "Error: Environment name is required. Use --list-envs to see available environments."
        )
        return

    # Apply robot type prefix if not sawyer
    env_name: str = args.env_name
    if args.arm_name != "sawyer":
        # Check if the environment exists for the requested robot type
        robot_prefix: str = ROBOT_ENV_PREFIX[args.arm_name]
        prefixed_env_name: str = f"{robot_prefix}{env_name}"

        # Verify this environment exists
        if prefixed_env_name in benchmark.train_classes:
            env_name = prefixed_env_name
        else:
            print(
                f"Warning: {prefixed_env_name} not found. Falling back to {env_name}."
            )
            print("The robot type might not be available for this environment.")

    # Set up environment
    try:
        if env_name in benchmark.train_classes:
            env_cls: Type[EnvType] = benchmark.train_classes[env_name]
            # Use 'rgb_array' render mode for headless operation
            env: EnvType = env_cls(
                render_mode="rgb_array",
                camera_name=args.camera_name,
            )
            env.done_on_success = args.policy_test
            env.ignore_termination = args.pygame_control or args.render_test

            # Find matching tasks
            matching_tasks: List[Task] = [
                t for t in benchmark.train_tasks if t.env_name == env_name
            ]

            if matching_tasks:
                # Check if the requested task index is valid
                task: Task
                if args.task < 0 or args.task >= len(matching_tasks):
                    print(
                        f"Warning: Task index {args.task} out of range. Using task 0."
                    )
                    task = matching_tasks[0]
                else:
                    task = matching_tasks[args.task]
                    print(
                        f"Using task {args.task} of {len(matching_tasks)} available tasks"
                    )

                env.set_task(task)
            else:
                print(f"Warning: No tasks found for {env_name}")

            # Set seeds if available
            try:
                env.seed(args.seed)
                env.action_space.seed(args.seed)
                env.observation_space.seed(args.seed)
            except AttributeError:
                print("Warning: Couldn't set all seeds.")
        else:
            print(f"Error: Environment '{env_name}' not found")
            sys.exit(1)
    except Exception as e:
        print(f"Error setting up environment: {e}")
        sys.exit(1)

    # Load policy if requested
    policy: Optional[PolicyType] = None
    if args.policy_test:
        # Use the base env name (without robot prefix) for policy lookup
        base_env_name: str = env_name
        for prefix in ROBOT_ENV_PREFIX.values():
            if prefix and env_name.startswith(prefix):
                base_env_name = env_name[len(prefix) :]
                break

        policy = get_policy(base_env_name, args.arm_name)
        if not policy:
            print(f"No policy available for {base_env_name}, using random actions")

    # Execute based on mode
    try:
        if args.pygame_control:
            if not PYGAME_AVAILABLE:
                print(
                    "Error: PyGame is not available. Please install pygame with 'pip install pygame'"
                )
                print("Running random policy instead.")
                trajectory = record_trajectory(
                    env, max_steps=args.steps, output_dir=output_dir
                )
            else:
                print(f"Starting PyGame control for {env_name}")
                keyboard_control(
                    env,
                    max_steps=args.steps,
                    output_dir=output_dir,
                    window_size=window_size,
                )
        elif args.policy_test:
            print(f"Running {'policy' if policy else 'random'} for {env_name}")
            trajectory: Dict[str, Any] = record_trajectory(
                env,
                policy=policy,
                max_steps=args.steps,
                loop=args.loop,
                output_dir=output_dir,
            )
            print(f"Trajectory completed. Success: {trajectory['success']}")
        elif args.render_test:
            print(f"Rendering {env_name} for {args.steps} steps")
            trajectory: Dict[str, Any] = record_trajectory(env, delay=args.delay)
            print(f"Trajectory completed. Success: {trajectory['success']}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        try:
            env.close()
            if PYGAME_AVAILABLE and pygame.get_init():
                pygame.quit()
        except:
            pass


if __name__ == "__main__":
    main()
