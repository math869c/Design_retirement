import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass, jit

from numba import njit

from consav.grids import nonlinspace
from consav.linear_interp import interp_2d, interp_3d

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
        par.beta   = 0.96    # discount factor
        par.sigma  = 3.0     # CRRA
        par.gamma  = 2.5    # labor disutility curvature
        par.mu     = 0.8
        par.a_bar  = 10.0

        par.r_a    = 0.02
        par.r_s    = 0.04
        par.tau    = 0.10    # 10% pension contribution
        par.chi    = 0.0     # public pension replacement
        par.delta  = 0.07    # human capital depreciation

        par.beta_1 = 0.08
        par.beta_2 = 0.00    # or a small positive number

        par.w_0    = 1.0

        ages       = np.arange(par.start_age, par.T + par.start_age + 1)
        par.pi     = 1 - ((ages - par.start_age) / (par.T + par.start_age - par.start_age))**2

        print(ages)
        print(par.pi)

        # Grids
        par.a_max  = 200
        par.a_min  = 0
        par.N_a    = 20

        par.s_max  = 200
        par.s_min  = 0
        par.N_s    = 20

        par.k_min  = 0
        par.k_max  = 200
        par.N_k    = 20

        par.h_min  = 0
        par.h_max  = np.inf

        par.c_min  = 0.001
        par.c_max  = np.inf


    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol
        sim = self.sim

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, 1.1)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, 1.1)
        par.k_grid = nonlinspace(par.k_min, par.k_max, par.N_k, 1.1)

        shape = (par.T, par.N_a, par.N_s, par.N_k)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        # Simulation
        par.simT = par.T # number of periods
        par.simN = 1_000 # number of individuals

        shape = (par.simN,par.simT)

        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.s = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)

        # e. initialization
        sim.a_init = np.zeros(par.simN)
        sim.s_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)



    def solve(self):

        par = self.par
        sol = self.sol

        for t in reversed(range(par.T)):
            print(f"We are in t = {t}")

            for a_idx, assets in enumerate(par.a_grid):
                for s_idx, savings in enumerate(par.s_grid):
                    for k_idx, human_capital in enumerate(par.k_grid):

                        idx = (t, a_idx, s_idx, k_idx)

                        if t == par.T - 1:
                            hours = 0

                            obj = lambda x: -self.value_last_period(x[0], assets)
                            init_c = 1
                            bounds = [self.budget_constraint(assets, hours, savings, human_capital, t)]
                            result = minimize(obj, init_c, bounds=bounds, method='L-BFGS-B')

                            sol.c[idx] = result.x[0]
                            sol.h[idx] = hours
                            sol.V[idx] = -result.fun

                        elif par.retirement_age <= t:
                            hours = 0

                            obj = lambda x: -self.value_function(x[0], hours, assets, savings, human_capital, t)
                            init_c = result.x[0]
                            bounds = [self.budget_constraint(assets, hours, savings, human_capital, t)]
                            result = minimize(obj, init_c, bounds=bounds, method='L-BFGS-B')

                            sol.c[idx] = result.x[0]
                            sol.h[idx] = hours
                            sol.V[idx] = -result.fun

                        else:
                            obj, consumption = lambda x: self.optimize_for_hours(x, assets, savings, human_capital, t)[0]

                            bounds = [self.budget_constraint(assets, hours, savings, human_capital, t), (par.h_min, par.h_max)]

                            if par.retirement_age - 1 == t:
                                init = np.array([0.5])
                            else:
                                init = np.array([result.x[0]])

                            result = minimize(obj, init, bounds=bounds, method='L-BFGS-B')

                            sol.c[idx] = consumption
                            sol.h[idx] = result.x[0]
                            sol.V[idx] = -result.fun


    def optimize_for_hours(self, h, a, s, k, t):
        par = self.par

        budget_constraint = self.budget_constraint(a, h, s, k, t)

        obj = lambda x: -self.value_function(x, h, a, s, k, t)

        init = par.c_min

        result = minimize(obj, init, bounds=[budget_constraint], method='L-BFGS-B')

        return result.fun, result.x[0]



    def budget_constraint(self, a, h, s, k, t):
        par = self.par

        if par.retirement_age + par.m <= t:
            return (par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + par.chi))
        
        elif par.retirement_age <= t < par.retirement_age + par.m:
            return (par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + (1/par.m)*s + par.chi))

        else:
            return (par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + (1-par.tau)*h*self.wage(k)))


    def utility(self, c, h):
        par = self.par

        return (c)**(1-par.sigma)/(1-par.sigma) - (h)**(1+par.gamma)/(1+par.gamma)
    
    def bequest(self, a):
        par = self.par

        return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)
    
    def wage(self, k):
        par = self.par

        return np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*k**2)
    
    def value_last_period(self, c, a):
        par = self.par
        h = 0

        a_next = (1+par.r_a)*a - c

        return self.utility(c, h) + self.bequest(a_next)
    


    def value_function(self, c, h, a, s, k, t):
        par = self.par
        sol = self.sol

        V_next = sol.V[t+1]
        
        if np.any(np.isnan(V_next)):
            print(f"V_next contains NaN at t = {t}")
            assert False

        if par.retirement_age + par.m <= t:
            a_next = (1.0+par.r_a)*a + par.chi - c
            s_next = 0
        
        elif par.retirement_age <= t < par.retirement_age + par.m:
            a_next = (1.0+par.r_a)*a + (1/par.m)*s + par.chi - c
            s_next = (1-1/par.m)*s

        else:
            a_next = (1.0+par.r_a)*a + (1-par.tau)*h*self.wage(k) - c
            s_next = (1+par.r_s)*s + par.tau*h*self.wage(k)

        k_next = (1-par.delta)*k + h

        V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid, V_next, a_next, s_next, k_next)

        return self.utility(c, h) + (1-par.pi[t+1])*par.beta*V_next_interp + par.pi[t+1]*self.bequest(a_next)
    

    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. loop over individuals and time
        for i in range(par.simN):

            # i. initialize states
            sim.a[i,0] = sim.a_init[i]
            sim.s[i,0] = sim.s_init[i]
            sim.k[i,0] = sim.k_init[i]

            for t in range(par.simT):

                # ii. interpolate optimal consumption and hours
                sim.c[i,t] = interp_3d(par.a_grid, par.k_grid, sol.c[t], sim.a[i,t], sim.k[i,t])
                sim.h[i,t] = interp_3d(par.a_grid, par.k_grid, sol.h[t], sim.a[i,t], sim.k[i,t])

                # iii. store next-period states
                if t<par.simT-1:
                    income = self.wage_func(sim.k[i,t],t)*sim.h[i,t]
                    sim.a[i,t+1] = (1+par.r)*(sim.a[i,t] + income - sim.c[i,t])
                    sim.k[i,t+1] = sim.k[i,t] + sim.h[i,t]