### coding: UTF-8
# MountainCarにおいて、Q学習を進めていき、
# その結果をgif出力する。
# Windowsで実行したのでimagemagickが必要。
# gif出力の参考：https://book.mynavi.jp/manatee/detail/id=88961

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 画像配列を渡して、それをgif画像にするメソッド
def display_frames_as_gif(frames,post_name):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),dpi=72.0)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),interval=20)
    anim.save('anim-'+str(post_name)+'.gif', writer='imagemagick')

class QLearn(object):
    def __init__(self, env, alpha, gamma, epsilon, digitize=40):
        self.env = env           # これがOpenAI Gymの環境
        self.alpha = alpha       # 学習率。どれだけQテーブルの値を変化させるか
        self.gamma = gamma       # 割引率。どれだけ未来の報酬を考慮するか
        self.epsilon = epsilon   # ランダムな方策を選ぶ割合
        self.digitize = digitize # 離散化した際の分割数

        self.n_actions = env.action_space.n  # とりうる行動の数
        self.q_table = np.zeros((digitize, digitize, self.n_actions)) # Q値を保存するテーブル。40,40,3を想定

        self.episode_count = 0

        self.frames=[]   # 学修の様子を画像として出力するための記録フレームリスト
        self.rewards=[]  # 報酬を記録するリスト

    # 観測結果がq_tableのどこか計算
    def observe_to_discrete (self, obs):
        _env_low = self.env.observation_space.low       # 位置と速度の最小値
        _env_high = self.env.observation_space.high     # 位置と速度の最大値
        _env_dx = (_env_high - _env_low) / self.digitize# 1領域が含む幅を計算

        pos = int( ( obs[0] - _env_low[0] ) / _env_dx[0] )
        vel = int( ( obs[1] - _env_low[1] ) / _env_dx[1] )
        # print(pos,vel,sep=" ",end = " -> ")
        return pos, vel

    # 行動の選択
    def act(self, obs, is_learn):
        if np.random.uniform(0, 1) > self.epsilon or not is_learn:
            pos, vel = self.observe_to_discrete(obs)        # greedyに選択
          #  print(pos,vel,sep=" " ,end = ".\n")
            _action = np.argmax(self.q_table[pos][vel])
        else:
            _action = np.random.choice(self.n_actions) # ランダムな探索
        return _action;

    # 1個先の未来を用いて現在の行動価値(Q)を更新する
    def update_q_table(self, now_obs, action, next_obs, reward, is_terminal):  # is_terminal???

        # 行動前の状態の行動価値 Q(s,a)
        _pos, _vel = self.observe_to_discrete(now_obs)
        _now_q_value = self.q_table[_pos][_vel][action]

        # 行動後の状態で得られる最大行動価値 Q(s',a')
        _next_pos, _next_vel = self.observe_to_discrete(next_obs)
        _next_max_q_value = max(self.q_table[_next_pos][_next_vel])

        # 行動価値関数の更新。ここスライドで説明
        self.q_table[_pos][_vel][action] = _now_q_value + self.alpha * (reward + self.gamma * _next_max_q_value - _now_q_value)

    # 1エピソード内のループ
    def try_episode(self,is_record):
        obs = self.env.reset() # 環境のリセットしたもので初期化
        episode_reward  = 0

        for step in range(200):
            if is_record :
                self.frames.append(env.render(mode='rgb_array'))  # frameに記録

            action = self.act(obs,is_learn = is_record)
            next_obs, reward, done, info = self.env.step(action)
            self.update_q_table(obs, action, next_obs, reward, done) # 現在の状態と得られた情報からQ値を更新
            obs = next_obs  #これを忘れると環境が更新されたことにならない
            episode_reward += reward

            # print(step,next_observation, reward, done, info , sep=" ") # 情報を出力するが、見たいなら何らかの条件をつけること。ログが荒れる。

            if done:    # doneがTrueになったら１エピソード終了
                if self.episode_count%100 == 0:

                    print('episode: {}, total_reward: {}'.format(self.episode_count, episode_reward))
                self.rewards.append(episode_reward)
                self.episode_count += 1
                break

        return episode_reward # 使い道はないが一応返しておく

    # 学習全体のループ
    def learn(self,episodes):

        for ep in range(episodes+1):
            # if(ep%2000 == 0):
            if False:  # gif画像で出力しないとき
                self.frames.clear()  # 記録リストをクリア
                self.try_episode(is_record = True)
                display_frames_as_gif(self.frames,ep)
            else:
                self.try_episode(is_record = False)
        self.env.close() # 学習が終わったので、環境も終了。

        # クラス内部に保存されてる学習結果とか出力するならここ
        path = './q_table.csv'
        with open(path, mode='w') as f:
            i = 0
            for x in self.q_table:
                f.write("\n")
                if i < 40:
                    f.write(str(x) + " , ")
                    i += 1
                else :
                    f.write("\n")
                    i = 0
            f.close()

        path = './reward.csv'
        with open(path, mode='w') as f:
            for x in self.rewards:
                f.write(str(x) + "\n")
            f.close()

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env = wrappers.Monitor(env, "./movie_folder", video_callable=(lambda ep: ep % 1000 == 0))


    model = QLearn(env, alpha=0.2, gamma=0.99, epsilon=0.002, digitize=40)
    model.learn(1000)
    env.close()
