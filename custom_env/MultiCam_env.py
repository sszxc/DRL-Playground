# Author: Xuechao Zhang
# Date: XXXX x th, 2022
# Description: 多相机主动监控 强化学习环境

from gym import core

class MultiCamEnv(core.Env):
    def __init__(self) -> None:
        super().__init__()
    
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError

if __name__ == '__main__':
    env = MultiCamEnv()
    observation = env.reset()
    print('State shape:', env.observation_space.shape)
    print('Number of actions:', env.action_space.n)