import gymnasium as gym
import numpy as np
import array
env = gym.make('CartPole-v1',render_mode='human')

# observation = env.reset()
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample() # your agent here (this takes random actions)
#     obs, reward, terminated, truncated, info = env.step(action)
#     print("observation",obs)
#     if terminated:
#         observation = env.reset()

# env.close()
# obs contain [-0.11021659 -0.60831076  0.07028293  0.7898158 ]
# change it to array using split function


def basic_policy(obs):
    if str(obs[2]) == ',':
        obs[2] = 0
    angle = float(str(obs[2]).replace(",",""))
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    
    episode_rewards = 0
    obs = env.reset()
    obs_str = str(obs)
    obs = obs_str.split()
 


   

    for step in range(200):
        action = basic_policy(obs)
        obs, reward, terminate,truncated, info = env.step(action)
        episode_rewards += reward
        if terminate:
            break
    totals.append(episode_rewards)

print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
