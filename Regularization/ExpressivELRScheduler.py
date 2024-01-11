
import math


class ExpressivELRScheduler:

    __alpha: float
    __min_alpha: float
    __decay: str
    __decay_rate: float

    __iteration = 0

    def __init__(self,
                 alpha: float,
                 min_alpha: float,
                 decay: str,
                 decay_rate: float):

        if alpha < 0 or min_alpha < 0:
            raise ValueError("Error: alpha must be greater than zero!")
        self.__alpha = alpha
        self.__min_alpha = min_alpha

        if decay != "none" and decay != "exponential" and decay != "inverse":
            raise ValueError("Error: only exponential, inverse and no decay are implemented!")
        self.__decay = decay
        self.__decay_rate = decay_rate

    def step(self):
        self.__iteration += 1

    def alpha(self):
        alpha = self.__alpha

        if self.__decay == "exponential":
            alpha = self.__alpha * math.exp(-self.__decay_rate * self.__iteration)
        elif self.__decay == "inverse":
            alpha = self.__alpha / (1 + self.__decay_rate * self.__iteration)

        return max(alpha, self.__min_alpha)