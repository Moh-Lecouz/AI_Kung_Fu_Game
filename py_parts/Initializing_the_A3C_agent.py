# Initializing the A3C agent

agent = Agent(number_actions)

# Evaluating our A3C agent on a single episode

def evaluate(agent, env, n_episodes = 1):
      episodes_rewards = []
      for _ in range(n_episodes):
          total_rewards = 0
          state, _ = env.reset()
          while True:
            action = agent.act(state)
            state, reward, done, info, _ = env.step(action[0])
            total_rewards += reward
            if done:
                break
          episodes_rewards.append(total_rewards)
      return episodes_rewards