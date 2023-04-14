import gymnasium as gym
import numpy as np
import array
env = gym.make('CartPole-v1',render_mode='human')



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


