class Experience:
    def __init__(self, state, preference_vector, action, new_state, reward):
        self.state = state
        self.preference_vector = preference_vector
        self.action = action
        self.new_state = new_state
        self.reward = reward

class ExperienceReplayBuffer:
    """Near-on Experience Replay Buffer from MOSEC paper"""
    def __init__(self, capacity=10000, similarity_threshold=0.1):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.similarity_threshold = similarity_threshold

    def add(self, state, preference_vector, action, new_state, reward):
        """Add experience to buffer"""
        experience = Experience(state, preference_vector, action, new_state, reward)
        self.buffer.append(experience)

    def sample(self):
        """Sample random experience from buffer"""
        if len(self.buffer) == 0:
            return None
        return random.choice(self.buffer)

    def size(self):
        return len(self.buffer)
