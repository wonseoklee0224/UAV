import random
import os
os.environ['TF_CPP_MIN_LEVEL'] = '2'

from random import randint
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from _collections import deque


import numpy as np
import envirionment

#이거추가한다. 변경사항을 인지할까?
gridSize = [200, 200, 200]
max_num_user = 30
number_user = 30
user_points = np.zeros((number_user,3))
for i in range(0, number_user):
    y = randint(0, gridSize[1])
    x = randint(0, gridSize[2])
    user_points[i][0] = 0
    user_points[i][1] = y
    user_points[i][2] = x
print(user_points)
EPISODES = 100
times = 15
MOS_array = np.zeros(EPISODES)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)
# print("추가됨~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def distance_cal(UAV_point, user_point):
    d = np.sqrt((UAV_point[0] - 0) ** 2 + (UAV_point[1] - user_point[1]) ** 2 + (UAV_point[2] - user_point[2]) ** 2)
    return d


def degree_cal(distance, UAV_H):
    degree_value = np.arcsin(UAV_H / distance) * 180 / np.pi
    return degree_value


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False
        # ??????????????????????????
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=2000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.update_target_model()

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            # dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            #  if dones[i]:
            #      target[i][actions[i]] = rewards[i]
            #   else:
            target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.model.save('model.h5')
        self.model.load_weights("model.h5")
        print("Saved model from disk")


if __name__ == "__main__":
    env = envirionment.Environment(gridSize, user_points)

    state_size = 31
    action_size = 7

    agent = DQNAgent(state_size, action_size)

    for x in range(EPISODES):
        tmp = 0
        result = 0

        uav_point = env.reset()
       # uav_coordinate = np.reshape(uav_coordinate, [1, 3])

        state = np.zeros((31, 1))

        for i in range(0, max_num_user):
            dis = distance_cal(uav_point, user_points[i])
            degree = degree_cal(dis, uav_point[0])
            print(dis)
            degree = int(degree/3)
            state[degree] += 1

        state = np.reshape(state, [1, 31])
        print(state)
        while tmp < times:
            next_state = np.zeros((31, 1))
            action = agent.get_action(state)
            reward, next_uav_point, MOS_array[x] = env.act(action)
            for j in range(0, max_num_user):
                dis = distance_cal(next_uav_point, user_points[j])
                degree = degree_cal(dis, next_uav_point[0])
                degree = int(degree / 3)
                next_state[degree] += 1
            next_state = np.reshape(next_state, [1, state_size])
            print(next_state)

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            agent.append_sample(state, action, reward, next_state)
            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
                result += reward
                state = next_state
                tmp += 1

        print("에피소드", x)
        print(MOS_array[x])
        print(state)
        print(next_uav_point)

    agent.save_model()
    x = np.arange(0, EPISODES, 1)
    plt.plot(x, MOS_array[x])
    plt.show()
