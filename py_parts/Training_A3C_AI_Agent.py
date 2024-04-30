# Training the A3C agent

import tqdm

env_batch = EnvBatch(number_actions)
batch_states = env_batch.reset()

with tqdm.trange(1, 3001) as progress_bar:
    for i in progress_bar:
        batch_actions = agent.act(batch_states)
        batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)
        batch_rewards *= 0.01
        agent.step(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
        batch_states = batch_next_states
        if i % 1000 == 0:
            print("Average agent reward:", np.mean(evaluate(agent, env, n_episodes = 10)))
