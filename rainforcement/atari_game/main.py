import gym

env_name='stocks-v0'
env = gym.make(env_name, render_mode='human')


obs = env.reset()
print(obs)
