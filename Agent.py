class Agent:
    def __init__(self):
        # Updated action dimension for ratio-based actions
        action_dim = len(ACTION_SPACE_RATIOS)

        self.QNet = MultiObjectiveDQN(state_dim=6, preference_dim=3, action_dim=action_dim).to(device)
        self.target_QNet = MultiObjectiveDQN(state_dim=6, preference_dim=3, action_dim=action_dim).to(device)

        # Copy weights to target network
        self.target_QNet.load_state_dict(self.QNet.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.QNet.parameters(), lr=0.001)

def epsilon_greedy(Q_values, epsilon):
    """
    Select an action using the ε-greedy strategy.
    """
    if random.random() < epsilon:
        # Explore: random action
        return random.randint(0, Q_values.shape[-1] - 1)
    else:
        # Exploit: best Q-value
        return Q_values.argmax().item()

# Updated action dimension
action_dim = len(ACTION_SPACE_RATIOS)
