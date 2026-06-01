class StoppingPolicy:
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def should_stop(self, state):
        return state.uncertainty < self.threshold