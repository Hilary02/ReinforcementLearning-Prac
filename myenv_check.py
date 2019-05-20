### coding: UTF-8
# 自作環境の動作チェック

import myenv
import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import os

class QLearn(object):
    def __init__(self, env, alpha, gamma, epsilon, digitize=40):
        self.env = env           # これがOpenAI Gymの環境
        self.alpha = alpha       # 学習率。どれだけQテーブルの値を変化させるか
        self.gamma = gamma       # 割引率。どれだけ未来の報酬を考慮するか
        self.epsilon = epsilon   # ランダムな方策を選ぶ割合
        # self.digitize = digitize # 離散化した際の分割数

        self.n_actions = env.action_space.n  # とりうる行動の数
        h,w = self.env.observation_space.shape
        self.q_table = np.zeros((h, w, self.n_actions)) # Q値を保存するテーブル。7,12,4を想定

        #shapeって何だろうか？みんな設定できる？

        self.episode_count = 0

        save_dir_env="dungeon"
        save_dir_param ="{}-a{}g{}e{}d{}".format(datetime.datetime.today().strftime("%y%m%d"),alpha,gamma,epsilon,digitize)
        self.save_dir=save_dir_env+"/"+save_dir_param+"/"
        self.frames=[]   # 学修の様子を画像として出力するための記録フレームリスト
        self.rewards=[]  # 報酬を記録するリスト

        for i in range(1,20):
            if os.path.isdir(self.save_dir) == False:
                os.makedirs(self.save_dir, exist_ok=True)
                break
            else:
                self.save_dir="{}-{}/".format(save_dir_env+"/"+save_dir_param,i)

    # 観測結果がq_tableのどこか計算
    def observe_to_discrete (self, obs):
        _h, _w = np.where(obs==6)
        # print("obs:",_h[0], _w[0])
        return _h[0], _w[0]

    # 行動の選択
    def act(self, obs, is_learn):
        if np.random.uniform(0, 1) > self.epsilon or not is_learn:
            posh, posw = self.observe_to_discrete(obs)        # greedyに選択
          #  print(pos,vel,sep=" " ,end = ".\n")
            _action = np.argmax(self.q_table[posh][posw])
        else:
            _action = np.random.choice(self.n_actions) # ランダムな探索
        return _action;

    # 1個先の未来を用いて現在の行動価値(Q)を更新する
    def update_q_table(self, now_obs, action, next_obs, reward, is_terminal):  # is_terminal???

        # 行動前の状態の行動価値 Q(s,a)
        posh, posw = self.observe_to_discrete(now_obs)        # greedyに選択
        _now_q_value = self.q_table[posh][posw][action]

        # 行動後の状態で得られる最大行動価値 Q(s',a')
        n_posh,n_posw = self.observe_to_discrete(next_obs)
        _next_max_q_value = max(self.q_table[n_posh][n_posw])

        # 行動価値関数の更新。ここスライドで説明
        self.q_table[posh][posw][action] = _now_q_value + self.alpha * (reward + self.gamma * _next_max_q_value - _now_q_value)

    # 1エピソード内のループ
    def try_episode(self,is_record):
        obs = self.env.reset() # 環境のリセットしたもので初期化
        episode_reward  = 0

        for step in range(100):
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
                break

        return episode_reward # 使い道はないが一応返しておく

    # 学習全体のループ
    def learn(self,episodes):

        for ep in range(episodes+1):
            self.episode_count += 1
            if(ep%2000 == 0):
            # if False:  # gif画像で出力しないとき
                self.frames.clear()  # 記録リストをクリア
                self.try_episode(is_record = False)
                # cant use
                # display_frames_as_gif(self.frames, ep, self.save_dir)
            else:
                self.try_episode(is_record = False)
        self.env.close() # 学習が終わったので、環境も終了。

        # クラス内部に保存されてる学習結果とか出力するならここ
        # note: csv出力に難あり

        path = self.save_dir+ 'q_table.csv'
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

        path = self.save_dir+ 'reward.csv'
        with open(path, mode='w') as f:
            for x in self.rewards:
                f.write(str(x) + "\n")
            f.close()

if __name__ == '__main__':
    env = gym.make('myenv-v0')

    # windows環境ではコメントアウトする
    # env = wrappers.Monitor(env, "./movie_folder", video_callable=(lambda ep: ep % 1000 == 0))

    model = QLearn(env, alpha=0.2, gamma=0.99, epsilon=0.002, digitize=40)
    model.learn(2000)
    env.close()
