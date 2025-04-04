import numpy as np
class Bernoulli:
    """ Simple Bernoulli random number generator """
    def __init__(self, p, size):
        """
        p: Probability of success (1)
        """
        self.p = np.asarray(p)
        self.size = size

    def rvs(self):
        """
        Draw Bernoulli(p) random variables.
        Returns an array of booleans if size != None,
        or a single boolean if size=None.
        """
        # Draw uniform random numbers with shape = size
        uniform_draws = np.random.uniform(size=self.size)

        # Ensure p can broadcast correctly:
        if self.p.ndim == 1 and len(self.size) == 2 and self.p.shape[0] == self.size[1]:
            # Reshape p to (1, T) so it broadcasts across N rows
            return uniform_draws < self.p.reshape(1, -1)
        else:
            # Let numpy handle other valid broadcasting cases
            return uniform_draws < self.p