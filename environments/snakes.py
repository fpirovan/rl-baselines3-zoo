from os.path import join as pjoin

import num2words
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from settings import BASE_DIR


class XSnakeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, pod_number=3):
        xml_name = f"Snake{self.get_env_num_str(pod_number)}.xml"
        xml_path = pjoin(BASE_DIR, "environments", "assets", xml_name)
        self.num_body = pod_number
        self._direction = 0
        self.ctrl_cost_coeff = 0.0001 / pod_number * 3

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
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


class YSnakeEnv(XSnakeEnv):
    def __init__(self, pod_number=3):
        XSnakeEnv.__init__(self, pod_number=pod_number)

    def reset_model(self):
        self._direction = 1
        return XSnakeEnv.reset_model(self)


class XSnakeThreeEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=3)


class XSnakeFourEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=4)


class XSnakeFiveEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=5)


class XSnakeSixEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=6)


class XSnakeSevenEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=7)


class XSnakeEightEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=8)


class XSnakeNineEnv(XSnakeEnv):
    def __init__(self):
        XSnakeEnv.__init__(self, pod_number=9)


class YSnakeThreeEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=3)


class YSnakeFourEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=4)


class YSnakeFiveEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=5)


class YSnakeSixEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=6)


class YSnakeSevenEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=7)


class YSnakeEightEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=8)


class YSnakeNineEnv(YSnakeEnv):
    def __init__(self):
        YSnakeEnv.__init__(self, pod_number=9)
