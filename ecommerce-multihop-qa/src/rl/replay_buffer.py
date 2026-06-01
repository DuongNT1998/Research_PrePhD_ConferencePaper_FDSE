class ReplayBuffer:

    def __init__(self):

        self.states = []

        self.actions = []

        self.rewards = []

        self.log_probs = []

        self.values = []

        self.dones = []

    def store(
        self,
        state,
        action,
        reward,
        log_prob,
        value,
        done
    ):

        self.states.append(state)

        self.actions.append(action)

        self.rewards.append(reward)

        self.log_probs.append(log_prob)

        self.values.append(value)

        self.dones.append(done)

    def clear(self):

        self.__init__()