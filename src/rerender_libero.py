import os
import logging
from tqdm import tqdm
from multiprocessing import Pool
from typing import Optional

import cv2
import h5py
import numpy as np

import robosuite.utils.transform_utils as T
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark, get_libero_path

RAW_DATA_PATH = os.path.join("data/LIBERO/v0")
NEW_DATA_PATH = os.path.join("data/LIBERO/v1")
os.makedirs(NEW_DATA_PATH, exist_ok=True)

def get_logger(
    filename: str,
    logger_name: Optional[str] = None,
    log_level: int = logging.INFO,
    fmt_str: str = "[%(asctime)s] [%(levelname)s] %(message)s",
    wirte_mode: str = "w"
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    handler = logging.FileHandler(filename, mode=wirte_mode)
    handler.setLevel(log_level)  # handler设置日志级别
    handler.setFormatter(logging.Formatter(fmt_str))  # handler对象设置格式
    logger.addHandler(handler)
    return logger


def depthimg2Meters(env, depth):
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent
    far = env.sim.model.vis.map.zfar * extent
    image = near / (1 - depth * (1 - near / far))
    return image

def get_task_map():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_names = os.listdir(RAW_DATA_PATH)
    task_map = {}
    task_lst = []
    for task_suite_name in task_suite_names:
        task_map[task_suite_name] = {}
        task_suite = benchmark_dict[task_suite_name]()
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            task_map[task_suite_name][task.name] = task_id
            task_lst.append([task_suite_name, task.name, task_id])
    return task_map, task_lst

task_map, task_lst = get_task_map()
task_lsts = {}
for task_suite_name, task_name, task_id in task_lst:
    if task_suite_name not in task_lsts.keys():
        task_lsts[task_suite_name] = [[]]
    if len(task_lsts[task_suite_name][-1]) == 10:
        task_lsts[task_suite_name].append([])
    task_lsts[task_suite_name][-1].append([task_suite_name, task_name, task_id])
values = task_lsts.values()
task_lsts = []
for val in values: task_lsts += val

def process_fn(task_lst, process_id):
    logger = get_logger(f"logs/{process_id}.log", logger_name=f"{process_id}", wirte_mode="a")
    with open(f"logs/{process_id}.txt", "a") as fo:
        benchmark_dict = benchmark.get_benchmark_dict()
        for task_suite_name, task_name, task_id in task_lst:
            try:
                logger.info(f"the task suite is: {task_suite_name}")
                logger.info(f"the task name is: {task_name}")
                logger.info(f"the task id is: {task_id}")
                task_suite = benchmark_dict[task_suite_name]()
                task = task_suite.get_task(task_id)
                task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
                env_args = {
                    "bddl_file_name": task_bddl_file,
                    "camera_heights": 128,
                    "camera_widths": 128,
                    "has_renderer": True,
                    "has_offscreen_renderer": False,
                    "control_freq": 20,
                    "camera_depths": True,
                }
                env = OffScreenRenderEnv(**env_args)
                env = env
                raw_dataset_path = os.path.join(RAW_DATA_PATH, task_suite_name, f"{task_name}_demo.hdf5")
                new_dataset_path = os.path.join(NEW_DATA_PATH, "rgb", task_suite_name, f"{task_name}_demo.hdf5")
                new_dataset_depth_path = os.path.join(NEW_DATA_PATH, "depth", task_suite_name, f"{task_name}_demo.hdf5")
                os.makedirs(os.path.dirname(new_dataset_path), exist_ok=True)
                os.makedirs(os.path.dirname(new_dataset_depth_path), exist_ok=True)
                with h5py.File(raw_dataset_path,"r") as source_file:
                    source_data = source_file["data"]
                    with h5py.File(new_dataset_path,"w") as target_file, h5py.File(new_dataset_depth_path,"w") as target_depth_file:
                        target_file.create_group("data")
                        target_depth_file.create_group("data")
                        for traj_idx, traj_name in enumerate(source_data.keys()):
                            logger.info(f"processing [{traj_idx+1}/{len(source_data.keys())}]")
                            env.seed(0)
                            env.reset()
                            init_state = source_data[traj_name]["states"][0]
                            obs = env.set_init_state(init_state)
                            actions = source_data[traj_name]["actions"]

                            new_rewards = []
                            new_agentview_depths = []
                            new_agentview_rgbs = []
                            new_eye_in_hand_depths = []
                            new_eye_in_hand_rgbs = []
                            new_dones = []
                            new_robot_states = []
                            new_ee_states = []
                            new_gripper_states = []
                            new_actions = []
                            for idx in range(len(actions)):
                                next_obs, reward, done, info = env.step(actions[idx])
                                agentview_depth = depthimg2Meters(env, obs["agentview_depth"])
                                eye_in_hand_depth = depthimg2Meters(env, obs["robot0_eye_in_hand_depth"])
                                # cv2.imshow("rgb", np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand_image"]], axis=1))
                                # cv2.waitKey(1)
                                new_rewards.append(reward)
                                new_agentview_depths.append(agentview_depth)
                                new_agentview_rgbs.append(obs["agentview_image"])
                                new_eye_in_hand_depths.append(eye_in_hand_depth)
                                new_eye_in_hand_rgbs.append(obs["robot0_eye_in_hand_image"])
                                new_dones.append(done)
                                new_robot_states.append(env.env.get_robot_state_vector(obs))
                                new_ee_states.append(
                                    np.hstack(
                                        (
                                            obs["robot0_eef_pos"],
                                            T.quat2axisangle(obs["robot0_eef_quat"]),
                                        )
                                    )
                                )
                                new_gripper_states.append(obs["robot0_gripper_qpos"])
                                new_actions.append(actions[idx])
                                obs = next_obs
                                if idx == len(actions) - 1: done = True
                                if done: 
                                    print(f"the traj {traj_name}, reward is {reward}, expected length is {len(actions)} real length is {len(new_actions)}")
                                    break
                            # cv2.destroyAllWindows()
                            target_file.create_dataset(f"data/{traj_name}/actions", data=new_actions)
                            target_file.create_dataset(f"data/{traj_name}/rewards", data=new_rewards)
                            target_file.create_dataset(f"data/{traj_name}/dones", data=new_dones)
                            target_file.create_dataset(f"data/{traj_name}/robot_states", data=np.stack(new_robot_states, axis=0))
                            target_file.create_dataset(f"data/{traj_name}/obs/agentview_rgb", data=new_agentview_rgbs)
                            target_depth_file.create_dataset(f"data/{traj_name}/obs/agentview_depth", data=new_agentview_depths)
                            target_file.create_dataset(f"data/{traj_name}/obs/eye_in_hand_rgb", data=new_eye_in_hand_rgbs)
                            target_depth_file.create_dataset(f"data/{traj_name}/obs/eye_in_hand_depth", data=new_eye_in_hand_depths)
                            target_file.create_dataset(f"data/{traj_name}/obs/ee_states", data=np.stack(new_ee_states, axis=0))
                            target_file.create_dataset(f"data/{traj_name}/obs/gripper_states", data=np.stack(new_gripper_states, axis=0)) 
                env.close()
            except Exception as e:
                fo.write(f"{task_suite_name}, {task_name}, {task_id}\n")

# pool = Pool(2)
for i, task_lst in enumerate(task_lsts):
    process_fn(task_lst, i)
    # pool.apply_async(process_fn, (task_lst, i))
# pool.close()
# pool.join()