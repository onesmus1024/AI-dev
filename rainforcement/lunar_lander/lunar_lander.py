import gymnasium as gym
environment_name = 'LunarLander-v2'
env = gym.make(environment_name, render_mode='human')
episodes = 10
for episode in range(1, episodes+1):
    state = env.reset()
    terminated = False
    score = 0

    while not terminated:
        env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

env.close()

