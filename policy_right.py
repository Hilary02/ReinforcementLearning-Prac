### coding: UTF-8
# MountainCarにおいて、常に右という方策を実行して、
# その結果をgif出力する。
# Windowsで実行したのでimagemagickが必要。
# gif出力の参考：https://book.mynavi.jp/manatee/detail/id=88961

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),dpi=90)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=20)

    anim.save('anim-right.gif', writer='imagemagick')
    #display(display_animation(anim, default_mode='loop'))

env = gym.make('MountainCar-v0')
observation = env.reset()

env.render()

print("Observe data High-low")
print(env.observation_space.high)
print(env.observation_space.low)

print("---")
ep = 1

frames=[]

# Windowsだと失敗した
# env = wrappers.Monitor(env, './movie_folder' , video_callable=(lambda ep: ep  == 1))

observation = env.reset()
for step in range(200):
    frames.append(env.render(mode='rgb_array'))  # frame
    # action = get_action(env, q_table, observation, episode)
    next_observation, reward, done, info = env.step(2)
    print(step,next_observation, reward, done, info , sep=" ")

    # observation = next_observation
env.close()

display_frames_as_gif(frames)


