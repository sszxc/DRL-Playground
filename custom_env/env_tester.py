import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # 在docker中加上这两句屏蔽图像输出
import sys
import gym
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.getcwd())  # 把当前工作路径添加到导包路径
from custom_env.Acrobot_env import MyAcrobotEnv

if __name__ == '__main__':
    # env = MyAcrobotEnv()
    env = gym.make('Acrobot-v1')
    observation = env.reset()
    for i in range(100):
        action = env.action_space.sample()

        starttime = datetime.now()
        observation, reward, done, info = env.step(action)
        endtime = datetime.now()
        print('Frame', i, 'Render time:', endtime - starttime)

        env.render()
        if done:
            break