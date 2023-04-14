from collections import deque
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers import Dense
import gymnasium as gym
from matplotlib.pyplot import plot as plt
env = gym.make('CartPole-v1',render_mode='human')
import random
import numpy as np




class DQLAgent:


    def __init__(self, gamma=0.95, hu=24, opt=Adam,
                lr=0.001, finish=False):


        self.finish = finish
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = gamma
        self.batch_size = 32
        self.max_treward = 0
        self.averages = list()
        self.memory = deque(maxlen=2000)
        self.osn = env.observation_space.shape[0]
        self.model = self._build_model(hu, opt, lr)

    def _build_model(self, hu, opt, lr):


        model = Sequential()
        model.add(Dense(hu, input_dim=self.osn,
                        activation='relu'))
        model.add(Dense(hu, activation='relu'))
        model.add(Dense(env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=opt(lr=lr))
        return model
    

    def act(self, state):


        if random.random() <= self.epsilon:
            return env.action_space.sample()
        action = self.model.predict(state)[0]
        return np.argmax(action)
    
    def replay(self):


        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(
                self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=1,
                        verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def learn(self, episodes):


        trewards = []
        for e in range(1, episodes + 1):
            state = env.reset() 
            state = np.reshape(state[0], [1, self.osn])
            for _ in range(5000):
                action = self.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = np.reshape(next_state,
                                        [1, self.osn])
                self.memory.append([state, action, reward,
                                    next_state, done])
                state = next_state
                if done:
                    treward = _ + 1
                    trewards.append(treward)
                    av = sum(trewards[-25:]) / 25
                    self.averages.append(av)
                    self.max_treward = max(self.max_treward, treward)
                    templ = 'episode: {:4d}/{} | treward: {:4d} | '
                    templ += 'av: {:6.1f} | max: {:4d}'
                    print(templ.format(e, episodes, treward, av,
                                    self.max_treward), end='\r')
                    break
            if av > 195 and self.finish:
                break
            if len(self.memory) > self.batch_size:
                self.replay()

    def test(self, episodes):


        trewards = []
        for e in range(1, episodes + 1):
            state = env.reset()

            for _ in range(5001):


                state = np.reshape(state[0], [1, self.osn])
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, terminated, truncated, info = env.step(
                    action)
                done = terminated or truncated
                state = next_state
                if done:
                    treward = _ + 1
                    trewards.append(treward)
                    print('episode: {:4d}/{} | treward: {:4d}'
                        .format(e, episodes, treward), end='\r')
                    break
        return trewards


episodes = 1000

agent = DQLAgent(finish=True)
# agent.learn(episodes)
agent.test(3)

plt.figure(figsize=(10, 6))
x = range(len(agent.averages))
y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
plt.plot(agent.averages, label='moving average')
plt.plot(x, y, 'r--', label='trend')
plt.xlabel('episodes')
plt.ylabel('total reward')
plt.legend()
