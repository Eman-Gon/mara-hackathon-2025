class MultiObjectiveDQN(nn.Module):
    """Multi-Objective Deep Q-Network - FIXED VERSION"""
    def __init__(self, state_dim=6, preference_dim=3, action_dim=9, hidden_dim=128):
        super(MultiObjectiveDQN, self).__init__()

        self.state_dim = state_dim
        self.preference_dim = preference_dim
        self.action_dim = action_dim

        # State processing network
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Preference processing network
        self.preference_net = nn.Sequential(
            nn.Linear(preference_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU()
        )

        # Combined processing network
        self.combined_net = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, preference_vector):
        """Forward pass through the network"""
        # Convert to tensors if numpy arrays
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
            if len(state.shape) == 1:
                state = state.unsqueeze(0)
        if isinstance(preference_vector, np.ndarray):
            preference_vector = torch.FloatTensor(preference_vector).to(device)
            if len(preference_vector.shape) == 1:
                preference_vector = preference_vector.unsqueeze(0)

        # Process state and preference separately
        state_features = self.state_net(state)
        preference_features = self.preference_net(preference_vector)

        # Combine features
        combined_features = torch.cat([state_features, preference_features], dim=-1)

        # Output Q-values
        q_values = self.combined_net(combined_features)

        return q_values
