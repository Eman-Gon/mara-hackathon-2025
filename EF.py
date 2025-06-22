
class EFMemory:
    """
    EFMemory
    ---------
    • record(ep, vec)  – save a (episode, preference_vector) pair
    • earliest()       – first vector ever stored  (smallest episode)
    • latest()         – most recently stored vector (largest episode)
    """
    def __init__(self):
        self.history = []  # [(episode, vector), …]

    def record(self, episode, vector):
        self.history.append((episode, vector))
        self.history.sort(key=lambda x: x[0])  # keep oldest first

    def earliest(self):
        return self.history[0][1] if self.history else generate_random_preference_vector()

    def latest(self):
        return self.history[-1][1] if self.history else generate_random_preference_vector()
