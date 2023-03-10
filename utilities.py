class LinearUtility:
    def __init__(
        self, alpha, beta, gamma,
    ):
        save__init__args(locals(), underscore=True)

    def reward(
        self, sales, waste, availability,
    ):
        return sales * self._alpha - waste * self._beta + availability * self._gamma
