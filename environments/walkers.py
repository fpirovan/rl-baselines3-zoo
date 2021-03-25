from os.path import join as pjoin

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from settings import BASE_DIR


class WalkersOstrichEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = pjoin(BASE_DIR, "environments", "assets", "WalkersOstrich.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (
                0.8 < height < 2.0 and
                -1.0 < ang < 1.0 and
                self.sim.data.site_xpos[0, 2] > 1.1
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20


class WalkersHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = pjoin(BASE_DIR, "environments", "assets", "WalkersHalfCheetah.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        alive_bonus = 1.0
        reward += alive_bonus
        s = self.state_vector()
        done = not (
                np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                self.sim.data.site_xpos[2, 2] > 1.2 and
                self.sim.data.site_xpos[0, 2] > 0.7 and
                self.sim.data.site_xpos[1, 2] > 0.7
        )
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class WalkersFullCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = pjoin(BASE_DIR, "environments", "assets", "WalkersFullCheetah.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run
        alive_bonus = 1
        reward += alive_bonus
        s = self.state_vector()
        done = not (
                np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                self.sim.data.site_xpos[0, 2] > 0.7 and
                self.sim.data.site_xpos[1, 2] > 0.7
        )
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq),
            self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class WalkersKangarooEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        xml_path = pjoin(BASE_DIR, "environments", "assets", "WalkersKangaroo.xml")

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt) / 2.0
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (
                0.8 < height < 2.0 and -1.0 < ang < 1.0 and
                0.8 < self.sim.data.site_xpos[0, 2] < 1.6
        )
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += 0.8
        self.viewer.cam.elevation = -20
