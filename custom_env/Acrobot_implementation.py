# Author: Yuhang Yan
# Date: April 1st, 2022
# Description: 倒立摆实现，含传统控制器，使用matplotlib输出

import numpy as np
from math import sin,cos
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Dynamics of acrobot: Double pendulum with actuated 2nd joint, while the 1nd joint is underactuated
def acrobot(x,t):
    # x = [q1,dq1,q2,dq2]
    '''   
          //
     \   //
      \ //        
        Θ         q2: the relative angle with link-1, which is negative while rotating clockwise
        \\  
         \\ |
           O|     q1: the absolute angle with the vertical line, which is negative while rotating clockwise
    ——————————————————
    '''
    # Constants
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 2.0
    lc1 = 0.5
    lc2 = 1.0
    J1 = 0.083
    J2 = 0.33
    g = 9.81

    a1 = J1 + m1*lc1*lc1 + m2*l1*l1
    a2 = J2 + m2*lc2*lc2
    a3 = m2*l1*lc2
    b1 = (m1*lc1 + m2*l1)*g
    b2 = m2*lc2*g

    #  M(q)
    M11 = a1 + a2 + 2*a3*cos(x[2])
    M12 = a2 + a3*cos(x[2])
    M21 = M12
    M22 = a2

    M = np.array([[M11,M12], [M21,M22]])


    #  H(q,dq)
    H = np.array([[a3*sin(x[2])*(-2*x[1]*x[3]-x[3]*x[3])],[a3*sin(x[2])*x[1]*x[1]]])

    #  G(q)
    G = np.array([[-b1*sin(x[0])-b2*sin(x[0]+x[2])],[-b2*sin(x[0]+x[2])]])

    # Angular Momentum
    L = (a1 + a2 + 2*a3 *cos(x[2]))*x[1]+(a2 + a3 *cos(x[2]))*x[3]
    dL = b1 *sin(x[0])+b2*sin(x[0]+x[2])
    ddL = b1*x[1] *cos(x[0])+b2*(x[1]+x[3]) *cos(x[0]+x[2])

    # Control gains
    kdd = 1.2534
    kd = 10.4721
    kp = 16.2728
    ks = -5.9899

    # Controller
    u = kdd*ddL+kd*dL+kp*L-ks*x[2]+(-b2 *sin(x[0]+x[2]))
    U = np.array([[0],[u]])

    ddq = np.matmul(np.linalg.inv(M),U - H - G).reshape(-1)
    dq = [x[1],x[3]]

    return [dq[0],ddq[0],dq[1],ddq[1]]

# Draw the animation
def animate(q1,q2):
    l1=1.0 # Lenth of links
    l2=2.0
    xorigin = 0
    yorigin = 0

    J1 = [xorigin,yorigin] # Location of the 1nd Joint
    scale = 1.0 # Change the size of marker/line

    fig, axis = plt.subplots()
    axis.grid()
    link1, = axis.plot([],[],'b.-',linewidth=scale*10,markersize=30*scale) # Line
    link2, = axis.plot([],[],'r.-',linewidth=scale*10,markersize=30*scale) # Line
    join1, = axis.plot([],[],'k.',markersize=30*scale) # Dot

    def init():
        xmin = xorigin - (l1+l2)
        xmax = xorigin + (l1+l2)
        ymin = yorigin - 1.5*l1
        ymax = yorigin + 1.5*(l1+l2)

        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        return link1, link2, join1

    def update(i):

        J2 = [J1[0]-l1*sin(q1[i]),J1[1]+l1*cos(q1[i])]
        Tip = [J1[0]-l1*sin(q1[i])-l2*sin(q1[i]+q2[i]),J1[1]+l1*cos(q1[i])+l2*cos(q1[i]+q2[i])]

        link1.set_data([J1[0],J2[0]],[J1[1],J2[1]])
        link2.set_data([J2[0],Tip[0]],[J2[1],Tip[1]])
        join1.set_data([J2[0]],[J2[1]])

        return link1, link2, join1

    # Animation, more details at https://blog.csdn.net/miracleoa/article/details/115407901
    # interval: the frequence of updating plot, unit: ms
    ani = animation.FuncAnimation(fig, update, range(0, len(q1)-1), init_func=init, interval=1)
    plt.show()

if __name__ == '__main__':

    pi = 3.1416
    x0 = [0.0, 0.0, -pi, 0.0]  # initial state, x0 = [q1,dq1,q2,dq2], final state = [0,0,0,0]
    t=np.linspace(0,10,200)
    x=odeint(acrobot,x0,t)

    animate(x[:,0],x[:,2])