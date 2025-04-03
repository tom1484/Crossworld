#!/usr/bin/env python3
"""
Simple environment render tester for MetaWorld.
Opens an environment, renders it, and steps through with zero actions.
Uses PyGame for visualization of environment status.

Usage:
    python render_tester.py --robot-type sawyer --env-name door-open-v2
    python render_tester.py --robot-type panda --env-name drawer-close-v2 --save-frames
    python render_tester.py --list-envs
    python render_tester.py --robot-type sawyer --env-name pick-place-v2 --task 3
"""

import traceback
import argparse
import time
import random
import sys
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type, TypeVar
import numpy as np
import metaworld
from metaworld import Task
from collections import OrderedDict
from datetime import datetime

# Add PyGame import for interactive visualization
try:
    import pygame
    from pygame.locals import QUIT, KEYDOWN

    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Type definitions
EnvType = TypeVar("EnvType")
BenchmarkType = Any  # metaworld.Benchmark

# Robot-specific environment prefix mapping
ARM_ENV_PREFIX: Dict[str, str] = {
    "sawyer": "",  # Sawyer is the default, no prefix needed
    "panda": "panda_",
    "jaco": "jaco_",
}


def get_environment_names(benchmark_class: BenchmarkType) -> List[str]:
    """Get all available environment names from a benchmark."""
    envs: List[str] = []
    for env_name in benchmark_class.train_classes.keys():
        envs.append(env_name)
    return sorted(envs)


def get_tasks_for_env(benchmark: BenchmarkType, env_name: str) -> List[Task]:
    """Get all tasks for a specific environment."""
    matching_tasks: List[Task] = [
        t for t in benchmark.train_tasks if t.env_name == env_name
    ]
    return matching_tasks


def save_render_frame(
    env: EnvType, output_dir: str, frame_num: int, prefix: str = "frame"
) -> Optional[str]:
    """Render the environment and save the frame as an image."""
    try:
        # Try different rendering approaches to handle API differences
        rgb_array: Optional[np.ndarray] = None

        # For MetaWorld environments, first try without any arguments
        try:
            rgb_array = env.render()
        except (TypeError, ValueError, AttributeError):
            pass

        # If that failed and we still don't have an image, try simple mode parameter
        if rgb_array is None:
            try:
                rgb_array = env.render(mode="rgb_array")
            except (TypeError, ValueError, AttributeError):
                pass

        # For some specific MetaWorld environments, try sim.render directly if available
        if rgb_array is None and hasattr(env, "sim"):
            try:
                rgb_array = env.sim.render(
                    camera_name="corner", width=500, height=500, depth=False
                )
            except (TypeError, ValueError, AttributeError):
                pass

        # Another variant for some MetaWorld environments
        if rgb_array is None and hasattr(env, "render_obs"):
            try:
                rgb_array = env.render_obs()
            except (TypeError, ValueError, AttributeError):
                pass

        if rgb_array is None:
            print(
                "Warning: Could not render the environment - all rendering methods failed"
            )
            return None

        # Save the image using PIL
        try:
            from PIL import Image

            filename: str = f"{prefix}_{frame_num:04d}.png"
            filepath: str = os.path.join(output_dir, filename)

            # Create Image from numpy array and save
            Image.fromarray(rgb_array).save(filepath)
            return filepath
        except ImportError:
            print("PIL not available, trying cv2...")
            try:
                import cv2

                filename: str = f"{prefix}_{frame_num:04d}.png"
                filepath: str = os.path.join(output_dir, filename)

                # OpenCV expects BGR format
                if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
                    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, bgr_array)
                else:
                    cv2.imwrite(filepath, rgb_array)
                return filepath
            except ImportError:
                print(
                    "WARNING: Neither PIL nor OpenCV available. Cannot save rendered frames."
                )
                return None
    except Exception as e:
        print(f"Error saving render frame: {e}")
        return None


def render_environment(
    env: EnvType,
    max_steps: int = 500,
    output_dir: Optional[str] = None,
    step_delay: float = 0.05,
    show_window: bool = True,
    window_size: Tuple[int, int] = (640, 480),
) -> None:
    """
    Render the environment and step through with zero actions.

    Args:
        env: The environment to render
        max_steps: Maximum number of steps to run
        output_dir: Directory to save rendered frames (None = no saving)
        step_delay: Time delay between steps in seconds
        show_window: Whether to try showing a window
        window_size: Size of the PyGame window (width, height)
    """
    # Reset the environment
    obs: np.ndarray = env.reset()
    print("Environment reset")

    # Create zero action
    action_dim: int = env.action_space.shape[0]
    zero_action: np.ndarray = np.zeros(action_dim)
    print(f"Using zero action with {action_dim} dimensions")

    # Initialize PyGame if requested to show window
    pygame_initialized = False
    screen = None

    if show_window and PYGAME_AVAILABLE:
        try:
            pygame.init()
            screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption("MetaWorld Environment Render")
            font = pygame.font.Font(None, 36)
            pygame_initialized = True
            print("PyGame window opened. Press Q to quit.")
        except Exception as e:
            print(f"Error initializing PyGame: {e}")
            pygame_initialized = False

    # Save initial frame
    if output_dir:
        frame_path = save_render_frame(env, output_dir, 0)
        if frame_path:
            print(f"Saved initial frame to: {frame_path}")

    # Step through the environment
    step_count: int = 0
    done: bool = False
    clock = pygame.time.Clock() if pygame_initialized else None
    running = True

    try:
        while running and step_count < max_steps and not done:
            # Check for PyGame quit events
            if pygame_initialized:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False
                        break
                    elif event.type == KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                            break

            if not running:
                break

            # Apply zero action
            next_obs, reward, terminated, truncated, info = env.step(zero_action)
            done = terminated or truncated

            # Print step information
            print(f"Step {step_count}/{max_steps}: Reward = {reward:.4f}")
            if "success" in info:
                print(f"Success: {info['success']}")

            # Render and possibly save frame
            rgb_array = None
            try:
                rgb_array = env.render()
            except Exception as e:
                print(f"Error rendering: {e}")

            if output_dir and rgb_array is not None:
                frame_path = save_render_frame(env, output_dir, step_count + 1)
                if frame_path:
                    print(f"Saved frame {step_count + 1} to: {frame_path}")

            # Update PyGame window if available
            if pygame_initialized and screen and rgb_array is not None:
                try:
                    # Convert numpy array to PyGame surface
                    rgb_array = (
                        np.transpose(rgb_array, (1, 0, 2))
                        if len(rgb_array.shape) == 3
                        else rgb_array
                    )
                    surface = pygame.surfarray.make_surface(rgb_array)
                    surface = pygame.transform.scale(surface, window_size)
                    screen.blit(surface, (0, 0))

                    # Draw status information
                    status_text = (
                        f"Step: {step_count}/{max_steps} | Reward: {reward:.4f}"
                    )
                    if "success" in info:
                        status_text += f" | Success: {info['success']}"

                    text_surface = font.render(status_text, True, (255, 255, 255))
                    text_bg = pygame.Surface(
                        (text_surface.get_width() + 20, text_surface.get_height() + 10),
                        pygame.SRCALPHA,
                    )
                    text_bg.fill((0, 0, 0, 128))
                    screen.blit(text_bg, (10, 10))
                    screen.blit(text_surface, (20, 15))

                    pygame.display.flip()

                except Exception as e:
                    print(f"Error updating PyGame window: {e}")

            step_count += 1
            obs = next_obs

            # Control the frame rate
            if pygame_initialized and clock:
                clock.tick(60)  # 60 FPS

            # Additional delay between steps if requested
            if step_delay > 0:
                time.sleep(step_delay)

            if done:
                print("Episode finished")
                # Reset if we want to continue after completion
                if step_count < max_steps:
                    obs = env.reset()
                    done = False

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Clean up
        if pygame_initialized:
            pygame.quit()
        try:
            env.close()
        except:
            pass

    print(f"Render test completed: {step_count} steps")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple renderer for MetaWorld environments"
    )
    parser.add_argument(
        "--arm-name",
        type=str,
        default="sawyer",
        choices=["sawyer", "panda", "jaco"],
        help="Robot arm type (sawyer, panda, jaco)",
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
        "--steps", type=int, default=500, help="Maximum number of steps to run"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save rendered frames (default: timestamped directory)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=False,
        help="Save rendered frames as images",
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
        "--delay",
        type=float,
        default=0.05,
        help="Delay between steps in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--show-window",
        action="store_true",
        default=False,
        help="Try to show a window directly (requires OpenCV)",
    )
    parser.add_argument(
        "--window-size",
        type=str,
        default="640,480",
        help="PyGame window size (width,height)",
    )

    args = parser.parse_args()

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set up output directory if saving frames
    # output_dir: Optional[str] = None
    # if args.save_frames:
    #     output_dir = args.output_dir
    #     if output_dir is None:
    #         timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         output_dir = os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)), f"render_output/{timestamp}"
    #         )

    #     if not os.path.exists(output_dir):
    #         try:
    #             os.makedirs(output_dir)
    #             print(f"Created output directory: {output_dir}")
    #         except Exception as e:
    #             print(f"Warning: Could not create output directory: {e}")
    #             output_dir = None

    # Parse window size
    try:
        width, height = map(int, args.window_size.split(","))
        window_size = (width, height)
    except:
        print("Warning: Invalid window size format. Using default 640x480.")
        window_size = (640, 480)

    # Initialize benchmark based on user's choice
    benchmark: Optional[BenchmarkType] = None
    benchmark_initializers: Dict[str, Callable[[], BenchmarkType]] = {
        "mt1": lambda: metaworld.MT1(
            env_name=args.env_name,
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "mt10": lambda: metaworld.MT10(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "mt50": lambda: metaworld.MT50(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "ml1": lambda: metaworld.ML1(
            env_name=args.env_name,
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "ml10": lambda: metaworld.ML10(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
        "ml45": lambda: metaworld.ML45(
            arm_name=args.arm_name,
            seed=args.seed,
        ),
    }

    # Try the user's selected benchmark first
    try:
        print(f"Initializing {args.benchmark.upper()} benchmark...")
        benchmark = benchmark_initializers[args.benchmark]()
        print(f"Successfully loaded {args.benchmark.upper()} benchmark")
    except Exception as first_error:
        print(f"Failed to load {args.benchmark.upper()} benchmark: {first_error}")
        print()
        
        traceback.print_exc()
        sys.exit(1)

    # List environments if requested
    if args.list_envs:
        env_names: List[str] = get_environment_names(benchmark)
        print("\nAvailable environments:")
        for env_name in env_names:
            arm_name: str = "sawyer"  # Default
            if env_name.startswith("panda_"):
                arm_name = "panda"
                base_name: str = env_name[6:]  # Remove panda_ prefix
            elif env_name.startswith("jaco_"):
                arm_name = "jaco"
                base_name = env_name[5:]  # Remove jaco_ prefix
            else:
                base_name = env_name

            print(f"  {env_name} (robot: {arm_name})")
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
        robot_prefix: str = ARM_ENV_PREFIX[args.arm_name]
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
            env: EnvType = env_cls(render_mode="rgb_array")

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

    # Render the environment
    print(f"\nStarting render test for {env_name}")
    print(f"Robot type: {args.arm_name}")
    print(f"Running for {args.steps} steps with {args.delay}s delay")
    print(f"Saving frames: {args.save_frames}")

    render_environment(
        env,
        max_steps=args.steps,
        # output_dir=output_dir,
        step_delay=args.delay,
        show_window=args.show_window,
        window_size=window_size,
    )


if __name__ == "__main__":
    main()
