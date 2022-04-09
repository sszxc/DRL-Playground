# Author: Xuechao Zhang
# Date: March 25th, 2022
# Description: 测试 TensorBoard
# Reference: https://www.cnblogs.com/amazingter/p/14666123.html

import sys, os
os.chdir(sys.path[0])  # 移动路径到当前文件所在位置
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs/test1")
# 画一幅图
for x in range(100):
    writer.add_scalar("y=x*x", x*x, x)

writer = SummaryWriter("./logs/test2")
# 在同一个section中画多幅图
for x in range(100):
    writer.add_scalar("function/y=x*x", x*x, x)
    writer.add_scalar("function/y=x+10", x+10, x)

writer = SummaryWriter("./logs/test3")
# 在同一个图中画多条曲线
for x in range(100):
    writer.add_scalars("Functions", {"y=x": x, "y=x+5": x+5}, x)
    writer.add_scalars("Functions", {"y=x+10": x+10}, x)

print("Done")