# Author: Xuechao Zhang
# Date: April 2nd, 2022
# Description: 魔改倒立摆模型以适用于Gym，但存在求解速度过慢问题

import numpy as np
from math import sin, cos
from scipy.integrate import odeint
from gym import core, spaces

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class MyAcrobotEnv(core.Env):
    def __init__(self):
        high = np.array([1.0, 1.0, 1.0, 1.0],
                        dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(low=low,
                                            high=high,
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(5, start=-2)  # {-2, -1, 0, 1, 2}
        self.next_action = 0

        # Constants
        self.m1 = 1.0
        self.m2 = 1.0
        self.l1 = 1.0
        self.l2 = 2.0
        lc1 = 0.5
        lc2 = 1.0
        J1 = 0.083
        J2 = 0.33
        g = 9.81

        self.a1 = J1 + self.m1 * lc1 * lc1 + self.m2 * self.l1 * self.l1
        self.a2 = J2 + self.m2 * lc2 * lc2
        self.a3 = self.m2 * self.l1 * lc2
        self.b1 = (self.m1 * lc1 + self.m2 * self.l1) * g
        self.b2 = self.m2 * lc2 * g

        # State
        self.pi = 3.1416
        self.state = [0.0, 0.0, -self.pi, 0.0]  # initial state [q1,dq1,q2,dq2]
        self.time_count = 0
        self.time_step = 0.05  # 控制周期

        self.fig, self.axis = plt.subplots()

    def reset(self):
        super(MyAcrobotEnv, self).reset()
        self.time_count = 0
        self.state = [0.0, 0.0, -self.pi, 0.0]
        return self.state

    def step(self, action):
        self.next_action = action * 6  # {-2, -1, 0, 1, 2}*6
        self.state = odeint(self.acrobot, self.state,
                        np.array([0, self.time_step]))[1]  # 更新状态
        q1 = self.state[0]
        q2 = self.state[2]
        reward = self.l1 * cos(q1) + self.l2 * cos(q1 + q2)  # 顶端高度计算reward
        self.time_count += 1
        terminal = False if self.time_count < 500 else True
        self.next_action = 0  # 归零动作
        return self.state, reward, terminal, {}

    def acrobot(self, x, t):
        '''
        Dynamics of acrobot:
        Double pendulum with actuated 2nd joint, while the 1nd joint is underactuated
        x = [q1,dq1,q2,dq2]
        ——————————————————
            //
        \   //
        \ //        
            Θ         q2: the relative angle with link-1, which is negative while rotating clockwise
            \\  
            \\ |
            O|     q1: the absolute angle with the vertical line, which is negative while rotating clockwise
        ——————————————————
        '''

        #  M(q)
        M11 = self.a1 + self.a2 + 2 * self.a3 * cos(x[2])
        M12 = self.a2 + self.a3 * cos(x[2])
        M21 = M12
        M22 = self.a2

        M = np.array([[M11, M12], [M21, M22]])

        #  H(q,dq)
        H = np.array([[self.a3 * sin(x[2]) * (-2 * x[1] * x[3] - x[3] * x[3])],
                    [self.a3 * sin(x[2]) * x[1] * x[1]]])

        #  G(q)
        G = np.array([[-self.b1 * sin(x[0]) - self.b2 * sin(x[0] + x[2])],
                    [-self.b2 * sin(x[0] + x[2])]])

        # Angular Momentum
        # L = (self.a1 + self.a2 + 2 * self.a3 * cos(x[2])) * x[1] + (self.a2 + self.a3 * cos(x[2])) * x[3]
        # dL = self.b1 * sin(x[0]) + self.b2 * sin(x[0] + x[2])
        # ddL = self.b1 * x[1] * cos(x[0]) + self.b2 * (x[1] + x[3]) * cos(x[0] + x[2])

        # # Control gains
        # kdd = 1.2534
        # kd = 10.4721
        # kp = 16.2728
        # ks = -5.9899

        # # Controller
        # u = kdd * ddL + kd * dL + kp * L - ks * x[2] + (-self.b2 * sin(x[0] + x[2]))
        # U = np.array([[0], [u]])
        # 控制器扭矩大约在 (-20, 20) 范围内

        u = self.next_action
        U = np.array([[0], [u]])

        ddq = np.matmul(np.linalg.inv(M), U - H - G).reshape(-1)
        dq = [x[1], x[3]]

        return [dq[0], ddq[0], dq[1], ddq[1]]

    def render(self, mode="rgb_array"):
        # Draw the animation
        def animate(q1,q2):
            xorigin = 0
            yorigin = 0

            J1 = [xorigin,yorigin] # Location of the 1nd Joint
            scale = 1.0 # Change the size of marker/line

            self.axis.grid()
            link1, = self.axis.plot([],[],'b.-',linewidth=scale*10,markersize=30*scale) # Line
            link2, = self.axis.plot([],[],'r.-',linewidth=scale*10,markersize=30*scale) # Line
            join1, = self.axis.plot([],[],'k.',markersize=30*scale) # Dot

            def init():
                xmin = xorigin - (self.l1+self.l2)
                xmax = xorigin + (self.l1+self.l2)
                ymin = yorigin - 1.5*self.l1
                ymax = yorigin + 1.5*(self.l1+self.l2)

                self.axis.set_xlim(xmin, xmax)
                self.axis.set_ylim(ymin, ymax)
                return link1, link2, join1

            def update(i):
                J2 = [J1[0]-self.l1*sin(q1[i]),J1[1]+self.l1*cos(q1[i])]
                Tip = [J1[0]-self.l1*sin(q1[i])-self.l2*sin(q1[i]+q2[i]),J1[1]+self.l1*cos(q1[i])+self.l2*cos(q1[i]+q2[i])]

                link1.set_data([J1[0],J2[0]],[J1[1],J2[1]])
                link2.set_data([J2[0],Tip[0]],[J2[1],Tip[1]])
                join1.set_data([J2[0]],[J2[1]])

                return link1, link2, join1

            # Animation, more details at https://blog.csdn.net/miracleoa/article/details/115407901
            # interval: the frequence of updating plot, unit: ms
            # ani = animation.FuncAnimation(fig, update, range(0, len(q1)-1), init_func=init, interval=1)

            # 直接渲染
            # ref: https://blog.csdn.net/chenxin0215/article/details/113770112
            init()
            update(0)
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            X = np.asarray(buf)
            plt.cla()
            return X

        return animate([self.state[0]], [self.state[2]])[...,0:3]  # 去掉透明度

if __name__ == "__main__":
    env = MyAcrobotEnv()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        env.step(action)
        print(action, env.state)