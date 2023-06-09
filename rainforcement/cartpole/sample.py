import gymnasium as gym
env = gym.make("CartPole-v1",render_mode='human')

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    print("action",action)
    observation, reward, terminated, truncated, info = env.step(0)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
