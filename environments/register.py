import num2words
import numpy as np
from gym.envs.registration import register

MAX_EPISODE_STEPS = 1000
TASKS = {
    "Centipede": [3, 5, 7] + [4, 6, 8, 10, 12] + [20, 40],
    "CpCentipede": [4, 6, 8, 10, 12],
    "XSnake": list(range(3, 10)) + [20, 40],
    "YSnake": list(range(3, 10))
}
CLASSES = {
    "Centipede": "centipedes",
    "CpCentipede": "centipedes",
    "Snake": "snakes",
    "BackSnake": "snakes"
}
THRESHOLDS = {
    "centipedes": 6000.0,
    "snakes": 360.0,
    "walkers": 3800.0
}


def register_all_envs():
    new_env_ids = []

    for env in TASKS:
        env_class = CLASSES[env]
        file_name = f"environments.{env_class}:"

        for i in np.sort(TASKS[env]):
            registered_name = env + num2words.num2words(i).capitalize()
            registered_name = registered_name.replace(" ", "")
            entry_point = file_name + registered_name + "Env"

            env_id = f"{registered_name}-v1"
            register(
                id=env_id,
                entry_point=entry_point,
                max_episode_steps=MAX_EPISODE_STEPS,
                reward_threshold=THRESHOLDS[env_class],
            )
            new_env_ids.append(env_id)

    env_id = "WalkersHalfHumanoid-v1"
    register(
        id=env_id,
        entry_point="environments.walkers:WalkersHalfHumanoidEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=THRESHOLDS["walkers"],
    )
    new_env_ids.append(env_id)

    env_id = "WalkersOstrich-v1"
    register(
        id=env_id,
        entry_point="environments.walkers:WalkersOstrichEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=THRESHOLDS["walkers"],
    )
    new_env_ids.append(env_id)

    env_id = "WalkersHopper-v1"
    register(
        id=env_id,
        entry_point="environments.walkers:WalkersHopperEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=THRESHOLDS["walkers"],
    )
    new_env_ids.append(env_id)

    env_id = "WalkersHalfCheetah-v1"
    register(
        id=env_id,
        entry_point="environments.walkers:WalkersHalfCheetahEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=THRESHOLDS["walkers"],
    )
    new_env_ids.append(env_id)

    env_id = "WalkersFullCheetah-v1"
    register(
        id=env_id,
        entry_point="environments.walkers:WalkersFullCheetahEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=THRESHOLDS["walkers"],
    )
    new_env_ids.append(env_id)

    env_id = "WalkersKangaroo-v1"
    register(
        id=env_id,
        entry_point="environments.walkers:WalkersKangarooEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        reward_threshold=THRESHOLDS["walkers"],
    )
    new_env_ids.append(env_id)

    return new_env_ids
