import numpy as np
class Bernoulli:
    """ Simple Bernoulli random number generator """
    def __init__(self, p):
        """
        p: Probability of success (1)
        """
        self.p = p

    def rvs(self, size=None):
        """
        Draw Bernoulli(p) random variables.
        Returns an array of booleans if size != None,
        or a single boolean if size=None.
        """
        return np.random.uniform(size=size) < self.p