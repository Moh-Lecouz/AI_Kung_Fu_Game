# Implementing the A3C class

class Agent():

    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.qnetwork = neural_network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork.parameters(), lr = learning_rate)

    def act(self, state):
        if state.ndim == 3:
            state = [state]
        state = torch.tensor(state, dtype = torch.float32, device = self.device)
        actions_values, _ = self.qnetwork(state)
        policy = F.softmax(actions_values, dim = -1)
        return np.array([np.random.choice(len(p), p = p) for p in policy.detach().cpu().numpy()])

    def step(self, state, action, reward, next_state, done):
        batch_size = state.shape[0]
        state = torch.tensor(state, dtype = torch.float32, device = self.device)
        next_state = torch.tensor(next_state, dtype = torch.float32, device = self.device)
        reward = torch.tensor(reward, dtype = torch.float32, device = self.device)
        done = torch.tensor(done, dtype = torch.bool, device = self.device).to(dtype = torch.float32)
        actions_values, state_value = self.qnetwork(state)
        _, next_state_value = self.qnetwork(next_state)
        target_state_value = reward + discount_factor * next_state_value * (1 - done)
        advantage = target_state_value - state_value
        probs = F.softmax(actions_values, dim = -1)
        logprobs = F.log_softmax(actions_values, dim = -1)
        entropy = -torch.sum(probs * logprobs, axis = -1)
        batch_idx = np.arange(batch_size)
        logp_actions = logprobs[batch_idx, action]
        actor_loss = - (logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(target_state_value.detach(), state_value)
        total_loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()