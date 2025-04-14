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

class Categorical:
    """ Categorical random number generator with time-varying probabilities. """
    def __init__(self, p, size=None):
        """
        p: Array-like of shape (k, T), where k = number of categories, T = time.
        size: Tuple like (N, T) specifying the number of samples per time step.
        """
        self.p = np.asarray(p)  # shape: (k, T)
        
        if self.p.ndim != 2:
            raise ValueError("p must be a 2D array of shape (categories, time)")

        if not np.allclose(np.sum(self.p, axis=0), 1):
            raise ValueError("Probabilities must sum to 1 along axis 0 (categories) for each time step")

        self.size = size  # expected to be (N, T)

    def rvs(self):
        """
        Returns an array of shape (N, T) of categorical samples in [0, k-1].
        """
        k, T = self.p.shape
        N, T_ = self.size

        if T != T_:
            raise ValueError("Time dimension in probabilities does not match requested size")

        samples = np.zeros((N, T), dtype=int)

        for t in range(T):
            # draw N samples for time t using the t-th column of p
            samples[:, t] = np.random.choice(k, size=N, p=self.p[:, t])

        return samples
