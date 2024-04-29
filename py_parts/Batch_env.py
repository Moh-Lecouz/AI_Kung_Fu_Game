# Testing multiple agents on multiple environments at the same time 

class EnvBatch():

    def __init__(self, n_envs = 10):
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        _states = []
        for envs in self.envs:
            _states.append(envs.reset()[0])
        return np.array(_states)

    def step(self, actions):
        next_states, rewards, dones, infos, _ = map(np.array, zip(*[envs.step(a) for envs, a in zip(self.envs, actions)]))
        for i in range(len(self.envs)):
            if dones[i]:
                next_states[i] = self.envs[i].reset()[0]
        return next_states, rewards, dones, infos