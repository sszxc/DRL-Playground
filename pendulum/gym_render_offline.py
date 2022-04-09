# Author: Xuechao Zhang
# Date: March 25th, 2022
# Description: 尝试 Docker 中的 Gym 环境, 绕过实时渲染，实现离线保存功能
# Reference: gym_to_gif https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553

from Acrobot_env import MyAcrobotEnv
from cv2 import imwrite, VideoWriter, VideoWriter_fourcc
import gym
from datetime import datetime, timedelta  # docker 中注意时区问题
import sys
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # 在docker中加上这两句屏蔽图像输出
os.chdir(sys.path[0])  # 移动路径到当前文件所在位置


class OfflineRenderer():
    def __init__(self):
        self.frames=[]
        self.naming_index = 1  # 保存图片起始索引

    def add(self, frame):
        self.frames.append(frame)

    def clear(self):
        '''
        清空图片序列
        '''
        self.frames=[]
        self.naming_index = 1

    def check_dirs(self, path):
        '''
        检查目录是否存在, 不存在则创建
        '''
        if not os.path.exists(path):
            os.makedirs(path)

    def export_image(self, image, naming="index", path="image/"):
        '''
        保存图片
        naming: time 按照时间命名; index 按照序号命名; 其他按照输入命名
        path: 图片保存路径, 默认为 image/
        '''
        self.check_dirs(path)
        filename = path
        if naming == "time":
            filename += (datetime.now()+timedelta(hours=8)).strftime('%H%M%S')
        elif naming == "index":
            filename += str(self.naming_index)
            self.naming_index += 1
        else:
            filename += naming
        filename += ".jpg"
        imwrite(filename, image)  # 非中文路径保存图片
        # cv2.imencode('.jpg', image)[1].tofile(filename)  # 中文路径保存图片
        print("image saved successfuly at", filename)

    def export_images(self, naming="index", path="image/"):
        '''
        保存图片序列
        naming: time 按照时间命名; index 按照序号命名; 其他按照输入命名
        path: 图片保存路径, 默认为 image/
        '''
        if len(self.frames) == 0:
            print("empty frames")
            return 0

        for image in self.frames:
            self.export_image(image, naming, path)

        print(len(self.frames), "images saved successfuly at", path)

        self.clear()

    def export_video(self, naming="time", format="mp4", path="video/"):
        '''
        保存图片序列到视频, 时间命名
        naming: 文件命名, 默认按照时间命名
        format: 视频格式, 支持 mp4 avi
        path: 图片保存路径, 默认为 video/
        '''
        if len(self.frames) == 0:
            print("empty frames")
            return 0

        self.check_dirs(path)

        frame_size = (self.frames[0].shape[1], self.frames[0].shape[0])
        fps = 20
        if naming == "time":  # 命名
            filename = (datetime.now()+timedelta(hours=8)).strftime('%H%M%S')
        else:
            filename = naming
        if format == "avi":  # 格式
            video_dir = os.path.join(path, filename + '.avi')
            fourcc = VideoWriter_fourcc(*'MJPG')
        elif format == "mp4":
            video_dir = os.path.join(path, filename + '.mp4')
            fourcc = VideoWriter_fourcc(*'mp4v')
        videowriter = VideoWriter(video_dir, fourcc, fps, frame_size)

        for frame in self.frames:  # 写入文件
            videowriter.write(frame)

        videowriter.release()

        print("video saved successfuly at", video_dir)
        self.frames=[]

if __name__ == '__main__':
    # 创造环境
    # env = gym.make('CartPole-v1')
    # env = gym.make('Acrobot-v1')
    # env = gym.make('Pendulum-v1')
    env = MyAcrobotEnv()

    observation = env.reset()  # 初始化环境 observation为环境状态

    print('State shape:', env.observation_space.shape)
    print('Number of actions:', env.action_space.n)

    renderer = OfflineRenderer()  # 收集环境渲染结果
    starttime = datetime.now()
    for t in range(100):
        action = env.action_space.sample()  # 随机采样动作空间的元素
        observation, reward, done, info = env.step(action)  # 与环境交互
        # print(observation, reward, done, info)
        if done:  # 使用环境的跳出条件
            print("Episode finished after {} timesteps".format(t + 1))
            break
        renderer.add(env.render(mode="rgb_array"))  # 图像引擎
    endtime = datetime.now()
    print('Render time:', endtime - starttime)

    renderer.export_video()  # 保存视频
    env.close()  # 关闭环境