class RewardFunction:
    def compute(self, state, answer_quality, step_cost):
        reward = 0

        # answer quality
        reward += answer_quality * 2.0

        # cost penalty
        reward -= step_cost * 0.1

        # uncertainty penalty
        reward -= state.uncertainty * 0.5

        return reward