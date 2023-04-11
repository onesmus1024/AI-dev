# import gym
# import numpy as np
# env = gym.make('CartPole-v1',render_mode='human')

# observation = env.reset()
# for _ in range(10):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     obs, reward, terminated, truncated, info = env.step(action)
#     print(obs)
#     if terminated:
#         observation = env.reset()

# env.close()

# def basic_policy(obs):
#     angle = obs[2]
#     return 0 if angle < 0 else 1

# totals = []
# for episode in range(500):
    
#     episode_rewards = 0
#     #obs = env.reset()
#     # for step in range(200):
#     #     action = basic_policy(obs)
#     #     obs, reward, terminate,truncated, info = env.step(action)
#     #     episode_rewards += reward
#     #     if terminate:
#     #         break
#     # totals.append(episode_rewards)

# print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

# import gym
# import numpy as np
# import pandas as pd
# np.random.seed(100)


# env = gym.make('CartPole-v1')

# action_size = env.action_space.n

# [env.action_space.sample() for _ in range(10)]


# state_size = env.observation_space.shape[0]
# state = env.reset()
# obs, reward, terminate, truncated, info = env.step(env.action_space.sample())


# data = pd.DataFrame()
# state = env.reset()
# length = []
# for run in range(25000):
#     done = False
#     prev_state = env.reset()
#     treward = 1
#     print("...")
#     results = []
#     while not done:
#         action = env.action_space.sample()
#         obs, reward, terminate, truncated, info  = env.step(action)
#         done = terminate
#         results.append({'s1': prev_state[0], 's2': prev_state[1],
                       
#                         'a': action, 'r': reward})
#         treward += reward if not done else 0
#         prev_state = state
#     if treward >= 110:
#         data = data.append(pd.DataFrame(results))
#         length.append(treward)


# np.array(length).mean()
# data.info()
# data.tail()


import gym
import os
import math
import random
import numpy as np
import pandas as pd
from pylab import plt, mpl
plt.style.use('seaborn')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(precision=4, suppress=True)
os.environ['PYTHONHASHSEED'] = '0'

env = gym.make('CartPole-v0')
env.seed(100)
env.action_space.seed(100)
env.observation_space
env.observation_space.low.astype(np.float16)
env.observation_space.high.astype(np.float16)
state = env.reset()
