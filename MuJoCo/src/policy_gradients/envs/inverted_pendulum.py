import numpy as np
import os

from gym import utils
from gym.envs.mujoco import mujoco_env
from gym.spaces import Box


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        mujoco_env.MujocoEnv.__init__(
            self,
            "%s/assets/inverted_pendulum.xml" % dir_path,
            2,
            **kwargs
        )
        
    def step(self, a):
        reward = 1.0
        self.do_simulation(a, self.frame_skip)

        ob = self._get_obs()
        terminated = bool(not np.isfinite(ob).all() or (np.abs(ob[1]) > 0.2))
        done = terminated
        # if self.render_mode == "human":
        #     self.render()
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent