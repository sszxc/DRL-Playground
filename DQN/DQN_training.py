# Author: Xuechao Zhang
# Date: March 25th, 2022
# Description: My first PyTorch Code
# Reference: 强化学习DQN 入门小游戏 最简单的Pytorch代码
#            https://blog.csdn.net/bu_fo/article/details/110871876

import gym
import torch
import random
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils.gym_render_offline import OfflineRenderer

from custom_env.Acrobot_env import MyAcrobotEnv  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNet(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 256)  # eg 从4个状态出发
        self.l2 = torch.nn.Linear(256, 128)
        self.l3 = torch.nn.Linear(128, output_size)  # eg 输出2个动作

    def forward(self, x):
        x = self.l1(x)
        x = torch.nn.functional.relu(self.l2(x))
        x = torch.nn.functional.relu(self.l3(x))
        return x

class Game:
    def __init__(self, config):
        # self.env = gym.make(config) #.unwrapped  # 解锁reward上限
        self.env = MyAcrobotEnv()  # 创造MyAcrobotEnv环境
        self.exp_pool = []                          # 经验池
        self.exp_pool_size = 100000                 # 经验池大小
        self.q_net = QNet(self.env.observation_space.shape[0],
                          self.env.action_space.n).to(device)  # Q网络
        self.explore = 0.2                          # 探索率 控制随机动作出现的频率
        self.explore_decay = 1e-7                   # 探索率衰减
        self.loss_fn = torch.nn.MSELoss().to(device)    # 损失函数
        self.opt = torch.optim.Adam(self.q_net.parameters())    # 优化器
        self.renderer = OfflineRenderer()           # Docker内 离线渲染视频
        self.log_interval = 10                      # 日志间隔
        self.renderer_interval = 2000               # 渲染间隔
        self.episode = -1                           # 训练轮次计数
        self.patch_size = 1000                      # 每一轮选用的样本数量
        self.name = config + '_' + datetime.now().strftime('%m.%d') + '_' \
            + (datetime.now() + timedelta(hours=8)).strftime('%H.%M')

    def __call__(self):
        self.writer = SummaryWriter("./log/" + self.name)

        while True:  # 开始一局游戏
            state = self.env.reset()    # 重置环境
            R = 0                       # 初始化奖励

            # 采集数据
            while True:
                if len(self.exp_pool) >= self.exp_pool_size:    # 经验池满了
                    self.exp_pool.pop(0)                            # 删除最旧的经验
                    self.explore -= self.explore_decay              # 探索率降低 随机动作减少
                    if torch.rand(1) > self.explore and self.episode % self.log_interval != 0:                        # 0-1随机数 判断是否随机动作
                        action = self.env.action_space.sample()
                    else:                                                   # 0-1随机数 用当前网络计算动作
                        _state = torch.tensor(state, dtype=torch.float32).to(device)       # 转成tensor格式
                        Qs = self.q_net(_state[None, ...])                      # 正向
                        action = torch.argmax(Qs, 1)[0].item()                  # 选择最大Q值的动作
                else:
                    action = self.env.action_space.sample()     # 经验池没满 随机动作


                # starttime = datetime.now()
                next_state, reward, done, _ = self.env.step(action)             # 与环境交互
                # endtime = datetime.now()
                # print('Called env.step, elapsed time:', endtime - starttime)

                R += reward                                                     # 本局游戏的总奖励
                # reward -= 1 * abs(next_state[0])  # 调整 CartPole reward
                # reward -= 5 * abs(next_state[2])  # 调整 CartPole reward
                # reward = (next_state[0] + 0.5)**2  # 调整 MountainCar reward
                self.exp_pool.append([state, reward, action, next_state, done]) # 存储到经验池
                state = next_state

                if self.episode % self.renderer_interval == 0:  # 添加到渲染器
                    self.renderer.add(self.env.render(mode="rgb_array"))

                if done:                            # 结束一轮训练（杆子倒了）
                    if self.episode % self.renderer_interval == 0:  # 渲染视频
                        self.renderer.export_video(naming = self.name + "_" + str(self.episode))
                        # self.renderer.clear()
                        self.save_net(self.name + "_" + str(self.episode))
                    if self.episode % self.log_interval == 0:  # 添加到日志
                        self.writer.add_scalar("Reward", R, self.episode * self.patch_size)  # 将loss写入tensorboard
                    break

            # 样本足够则开始训练，否则继续采数据
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=self.patch_size)            # 随机选择经验
                _state = torch.tensor(np.array(
                    [exp[0] for exp in exps])).to(device).float()  # 将经验解码成tensor格式
                _reward = torch.tensor(np.array([exp[1]
                                        for exp in exps])).to(device).unsqueeze(1)
                _action = torch.tensor(np.array([exp[2]
                                        for exp in exps])).to(device).unsqueeze(1)
                _next_state = torch.tensor(np.array([exp[3]
                                            for exp in exps])).to(device).float()
                _done = torch.tensor(np.array([int(exp[4]) for exp in exps
                                               ])).to(device).unsqueeze(1)

                # 预测值
                _Qs = self.q_net(_state)                                # 从state正向算两种action的Q值
                _Q = torch.gather(_Qs, 1, _action)                      # 根据sample选择的action得到Q
                # 目标值
                _next_Qs = self.q_net(_next_state)                      #
                _max_Q = torch.max(_next_Qs, dim=1, keepdim=True)[0]    # 在第二个维度上求最大值
                _target_Q = _reward.to(
                    torch.float32) + (1 - _done) * 0.9 * _max_Q  # 调整reward之后修改一下数据结构

                loss = self.loss_fn(_Q, _target_Q.detach())     # 计算损失
                self.opt.zero_grad()                            # 清空梯度
                loss.backward()                                 # 反向传播
                self.opt.step()                                 # 更新网络参数

                self.episode += 1

    def save_net(self, name):
        torch.save(self.q_net.state_dict(), "./model/" + name + ".pth")
        print("model saved successfuly at model/" + name + ".pth")

    def load_net(self, name):
        self.q_net.load_state_dict(torch.load("./model/" + name + ".pth"))  # not tried

    def play_game(self):
        '''
        测试一局游戏
        '''
        state = self.env.reset()
        R = 0
        while True:
            _state = torch.tensor(state, dtype=torch.float32).to(device)
            Qs = self.q_net(_state[None, ...])
            action = torch.argmax(Qs, 1)[0].item()

            next_state, reward, done, _ = self.env.step(action)
            R += reward
            state = next_state

            self.renderer.add(self.env.render(mode="rgb_array"))
            if done:
                self.renderer.export_video(naming=self.name + "_play")
                print("play game reward:", R)
                break

if __name__ == '__main__':
    # g = Game('CartPole-v1')
    # g = Game('MountainCar-v0')
    g = Game('MyAcrobot')
    g()
    # g.load_net("CartPole-v1_03.28_21.47_30000")
    # g.play_game()