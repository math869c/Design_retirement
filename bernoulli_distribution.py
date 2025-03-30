import numpy as np
#created a bernoulli class

class Bernoulli:
    def __init__(self, p):
        self.p = p

    def pmf(self, x):
        return self.p**x * (1 - self.p)**(1 - x)

    def mean(self):
        return self.p

    def var(self):
        return self.p * (1 - self.p)

    def std(self):
        return self.var()**0.5
    
    def rvs(self, shape=(1, 1)):
        return (np.random.rand(*shape) < self.p).astype(int)