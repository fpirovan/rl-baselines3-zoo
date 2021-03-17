from os.path import join as pjoin

import num2words
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from settings import BASE_DIR


class SnakeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, pod_number=3):
        xml_name = "Snake" + self.get_env_num_str(pod_number) + ".xml"
        xml_path = pjoin(BASE_DIR, "environments", "assets", xml_name)
        self.num_body = pod_number
        self._direction = 0
        self.ctrl_cost_coeff = 0.0001 / pod_number * 3

        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    @staticmethod
    def get_env_num_str(number):
        return num2words.num2words(number).capitalize()

    def step(self, a):
        xposbefore = self.sim.data.site_xpos[0][self._direction]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.site_xpos[0][self._direction]
        reward_fwd = (xposafter - xposbefore) / self.dt
        reward_ctrl = - self.ctrl_cost_coeff * np.square(a).sum()
        reward = reward_fwd + reward_ctrl
        ob = self._get_obs()
        return ob, reward, False, dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(
                low=-0.1, high=0.1, size=self.model.nq
            ),
            self.init_qvel + self.np_random.uniform(
                low=-0.1, high=0.1, size=self.model.nv
            )
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 1.2
        body_name = "podBody_" + str(int(np.ceil(self.num_body / 2)))
        self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)


class BackSnakeEnv(SnakeEnv):
    def __init__(self, pod_number=3):
        SnakeEnv.__init__(self, pod_number=pod_number)

    def reset_model(self):
        self._direction = 1
        SnakeEnv.reset_model(self)


class SnakeTwentyEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=20)


class SnakeFortyEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=40)


class SnakeThreeEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=3)


class SnakeFourEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=4)


class SnakeFiveEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=5)


class SnakeSixEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=6)


class SnakeSevenEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=7)


class SnakeEightEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=8)


class SnakeNineEnv(SnakeEnv):
    def __init__(self):
        SnakeEnv.__init__(self, pod_number=9)


class BackSnakeThreeEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=3)


class BackSnakeFourEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=4)


class BackSnakeFiveEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=5)


class BackSnakeSixEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=6)


class BackSnakeSevenEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=7)


class BackSnakeEightEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=8)


class BackSnakeNineEnv(BackSnakeEnv):
    def __init__(self):
        BackSnakeEnv.__init__(self, pod_number=9)
