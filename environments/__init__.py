from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3

from environments.register import register_all_envs

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3
}
