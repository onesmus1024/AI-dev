import gym
import tensorflow as tf
import numpy as np
import random
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score

env = gym.make('CartPole-v1')


class NNAgent:
    def __init__(self):
        self.max = 0
        self.scores = list()
        self.memory = list()
        self.model = self._build_model()

    def _build_model(self):

        model = Sequential()
        model.add(Dense(24, input_dim=4,
                        activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=RMSprop(lr=0.001))
        return model

    def act(self, state):
        if random.random() <= 0.5:
            return env.action_space.sample()
        action = np.where(self.model.predict(
            state, batch_size=None)[0, 0] > 0.5, 1, 0)
        return action

    def train_model(self, state, action):

        self.model.fit(state, np.array([action,]),
                       epochs=1, verbose=False)

    def learn(self, episodes):
        for e in range(1, episodes + 1):
            state = env.reset()
            for _ in range(201):
                state_mod = np.reshape([0, 7, 7, 7], [1, 4])

                action = self.act(state_mod)

                next_state, reward, done, truncated, info = env.step(
                    action)
                if done:
                    score = _ + 1
                    self.scores.append(score)
                    self.max = max(score, self.max)
                    print('episode: {:4d}/{} | score: {:3d} | max: {:3d}'
                          .format(e, episodes, score, self.max), end='\r')
                    break
                self.memory.append((state, action))
                self.train_model(state_mod, action)
                state = next_state


agent = NNAgent()
episodes = 500
agent.learn(episodes)
sum(agent.scores) / len(agent.scores)
