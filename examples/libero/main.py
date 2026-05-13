import collections
import dataclasses
import json
import logging
import math
import pathlib
from typing import Tuple

import imageio
try:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError:
    from libero import benchmark
    from libero import get_libero_path
    from libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_ids: Tuple[int, ...] = ()
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = ""

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    task_ids = list(args.task_ids) if args.task_ids else list(range(num_tasks_in_suite))
    invalid_task_ids = [task_id for task_id in task_ids if task_id < 0 or task_id >= num_tasks_in_suite]
    if invalid_task_ids:
        raise ValueError(f"Invalid task ids for {args.task_suite_name}: {invalid_task_ids}")
    logging.info(f"Evaluating task ids: {task_ids}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    per_task_results = []
    for task_id in tqdm.tqdm(task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            done = False
            replay_images = []
            # Track trajectory and action chunk boundaries
            ee_positions = []        # ee_pos at each active step
            chunk_boundaries = []    # step indices where a new action chunk starts
            chunk_full_actions = []  # full predicted action chunks (all 10 actions)
            active_step = 0          # counter for steps after num_steps_wait

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)
                    # Record end-effector position
                    ee_positions.append(obs["robot0_eef_pos"].tolist())

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        chunk_boundaries.append(active_step)

                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                        chunk_full_actions.append([a.tolist() for a in action_chunk])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    active_step += 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        # Record final position after last action
                        ee_positions.append(obs["robot0_eef_pos"].tolist())
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_stem = f"rollout_{args.task_suite_name}_task{task_id:02d}_{episode_idx:03d}_{task_segment}_{suffix}"
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"{video_stem}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Save trajectory data alongside the video
            traj_data = {
                "task_suite": args.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "success": bool(done),
                "replan_steps": args.replan_steps,
                "action_horizon": len(chunk_full_actions[0]) if chunk_full_actions else 0,
                "num_steps": len(ee_positions),
                "ee_positions": ee_positions,
                "chunk_boundaries": chunk_boundaries,
                "chunk_full_actions": chunk_full_actions,
                "fps": 10,
            }
            traj_path = pathlib.Path(args.video_out_path) / f"{video_stem}.json"
            with open(traj_path, "w") as tf:
                json.dump(traj_data, tf)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes)
        per_task_results.append(
            {
                "task_suite_name": args.task_suite_name,
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
        )
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    if args.results_out_path:
        results_path = pathlib.Path(args.results_out_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as file:
            json.dump(
                {
                    "task_suite_name": args.task_suite_name,
                    "task_ids": task_ids,
                    "num_trials_per_task": args.num_trials_per_task,
                    "seed": args.seed,
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                    "total_success_rate": float(total_successes) / float(total_episodes),
                    "per_task": per_task_results,
                },
                file,
                indent=2,
            )


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
