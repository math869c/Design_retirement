import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # Unpack
        par = self.par

        par.start_age = 30  # Time when agents enter the workforce
        par.retirement_age = 65 - par.start_age # Time when agents enter pension

        par.T = 100 - par.start_age # time periods

        par.m = 10 # Years with retirement payments

        # Preferences
        par.beta = 0.98
        par.sigma = 0.9
        par.gamma = 0.5
        par.mu = 0.8
        par.a_bar = 10

        par.r_a = 0.02
        par.tau = 0.4
        par.chi = 10
        par.r_s = 0.03

        # Grids
        par.a_max = 100
        par.a_min = 0
        par.N_a = 50

        par.s_max = 100
        par.s_min = 0
        par.N_s = 50


        par.h_min = 0
        par.h_max = np.inf

        par.c_min = 0.001
        par.c_max = np.inf


    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, 1.1)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, 1.1)

        shape = (par.T,par.N_a,par.N_s)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

    def solve(self):

        par = self.par
        sol = self.sol

        for t in reversed(range(par.T)):
            print(f"We are in t = {t}")

            for a_idx, assets in enumerate(par.a_grid):
                for s_idx, savings in enumerate(par.s_grid):

                    idx = (t, a_idx, s_idx)

                    if t == par.T - 1:
                        hours = 0

                        obj = lambda x: -self.value_last_period(x[0], assets)
                        init_c = 1
                        bounds = [(par.c_min, par.c_max)]
                        result = minimize(obj, init_c, bounds=bounds, method='L-BFGS-B')

                        sol.c[idx] = result.x[0]
                        sol.h[idx] = hours
                        sol.V[idx] = -result.fun

                    elif par.retirement_age <= t:
                        hours = 0

                        obj = lambda x: -self.value_function(x[0], hours, assets, savings, t)
                        init_c = result.x[0]
                        bounds = [(par.c_min, par.c_max)]
                        result = minimize(obj, init_c, bounds=bounds, method='L-BFGS-B')

                        sol.c[idx] = result.x[0]
                        sol.h[idx] = hours
                        sol.V[idx] = -result.fun

                    else:
                        obj = lambda x: -self.value_function(x[0], x[1], assets, savings, t)

                        bounds = [(par.c_min, par.c_max), (par.h_min, par.h_max)]

                        if par.retirement_age - 1 == t:
                            init = np.array([result.x[0], 0.5])
                        else:
                            init = np.array([result.x[0], result.x[1]])

                        result = minimize(obj, init, bounds=bounds, method='L-BFGS-B')

                        sol.c[idx] = result.x[0]
                        sol.h[idx] = result.x[1]
                        sol.V[idx] = -result.fun


    def utility(self, c, h):
        par = self.par

        return (c)**(1-par.sigma)/(1-par.sigma) - (h)**(1-par.gamma)/(1-par.gamma)
    
    def bequest(self, a):
        par = self.par

        return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)
    
    def value_last_period(self, c, a):
        par = self.par
        h = 0

        a_next = (1+par.r_a)*a - c

        return self.utility(c, h) + self.bequest(a_next)
    
    def value_function(self, c, h, a, s, t):
        par = self.par
        sol = self.sol

        V_next = sol.V[t+1]

        if par.retirement_age + par.m <= t:
            a_next = (1.0+par.r_a)*a + par.chi - c
            s_next = 0
        
        elif par.retirement_age <= t < par.retirement_age + par.m:
            a_next = (1.0+par.r_a)*a + (1/par.m)*s + par.chi - c
            s_next = (1-1/par.m)*s

        else:
            a_next = (1.0+par.r_a)*a + (1-par.tau)*h*100- c
            s_next = (1+par.r_s)*s + par.tau*h

        V_next_interp = interp_2d(par.a_grid,par.s_grid,V_next,a_next,s_next)

        return self.utility(c, h) + par.beta*V_next_interp + self.bequest(a)