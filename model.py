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

        # Optimization settings
        par.opt_method = 'L-BFGS-B'
        par.opt_tol = 1e-6
        par.opt_maxiter = 1000


        # Time
        par.start_age = 30  # Time when agents enter the workforce
        par.retirement_age = 97 - par.start_age # Time when agents enter pension
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
        par.H      = 0.1
 
        par.tau    = 0.10    # 10% pension contribution
        par.chi    = 0.0     # public pension replacement
        par.delta  = 0.07    # human capital depreciation

        par.beta_1 = 0.001
        par.beta_2 = 0.001    # or a small positive number

        par.w_0    = 1.0

        ages       = np.arange(par.start_age, par.T + par.start_age + 1)
        par.pi     = 1 - ((ages - par.start_age) / (par.T + par.start_age - par.start_age))**2

        # Grids
        par.a_max  = 200
        par.a_min  = 0
        par.N_a    = 30

        par.s_max  = 200
        par.s_min  = 0
        par.N_s    = 30

        par.k_min  = 0
        par.k_max  = par.retirement_age - par.start_age
        par.N_k    = 15

        par.h_min  = 0
        par.h_max  = 1

        par.c_min  = 0.001
        par.c_max  = np.inf


        par.stop_parameter = 0


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
        par.simN = 1 # number of individuals

        shape = (par.simN,par.simT)

        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.s = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.w = np.nan + np.zeros(shape)

        # e. initialization
        sim.a_init = np.ones(par.simN)*par.H
        sim.s_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.w_init = np.ones(par.simN)*par.w_0



    def solve(self):

        par = self.par
        sol = self.sol

        for t in reversed(range(par.T)):
            print(f"We are in t = {t}")
            par.stop_parameter = 0

            for a_idx, assets in enumerate(par.a_grid):
                for s_idx, savings in enumerate(par.s_grid):
                    for k_idx, human_capital in enumerate(par.k_grid):

                        idx = (t, a_idx, s_idx, k_idx)

                        if t == par.T - 1:
                            hours = 0

                            obj = lambda x: -self.value_last_period(x[0], assets)

                            bc_min, bc_max = self.budget_constraint(assets, hours, savings, human_capital, t)
                            bounds = [(bc_min, bc_max)]
                            
                            init_c = par.c_min
                            result = minimize(obj, init_c, bounds=bounds, method=par.opt_method, tol=par.opt_tol, options={'maxiter':par.opt_maxiter})

                            sol.c[idx] = result.x[0]
                            sol.h[idx] = hours
                            sol.V[idx] = -result.fun

                        elif par.retirement_age <= t:
                            hours = 0

                            obj = lambda x: -self.value_function(x[0], hours, assets, savings, human_capital, t)

                            bc_min, bc_max = self.budget_constraint(assets, hours, savings, human_capital, t)
                            bounds = [(bc_min, bc_max)]

                            init_c = np.min([result.x[0], bc_max])
                            result = minimize(obj, init_c, bounds=bounds, method=par.opt_method, tol=par.opt_tol, options={'maxiter':par.opt_maxiter})

                            sol.c[idx] = result.x[0]
                            sol.h[idx] = hours
                            sol.V[idx] = -result.fun

                        else:
                            init_c = sol.c[(t+1, a_idx, s_idx, k_idx)]

                            if par.retirement_age - 1 == t:
                                init_c = par.c_min
                                init_h = par.h_max
                            else:
                                init_c = result.x[0]
                                init_h = result.x[1]

                            obj = lambda x: -self.value_function(x[0], x[1], assets, savings, human_capital, t)

                            init = [init_c, init_h]

                            def constraint_func(x):
                                c, h = x
                                return self.budget_constraint(assets, h, savings, human_capital, t)[1] - c

                            constraints = [{'type':'ineq', 'fun': constraint_func}]

                            bounds = [(par.c_min, par.c_max), (par.h_min, par.h_max)]
                            result = minimize(obj, init, bounds=bounds, constraints=constraints, method=par.opt_method, tol=par.opt_tol, options={'maxiter':par.opt_maxiter})

                            sol.c[idx] = result.x[0]
                            sol.h[idx] = result.x[1]
                            sol.V[idx] = -result.fun


    # def optimize_consumption(self, h, a, s, k, init, t):

    #     bc_min, bc_max = self.budget_constraint(a, h, s, k, t)
    #     bounds = [(bc_min, bc_max)]
       
    #     obj = lambda x: -self.value_function(x[0], h, a, s, k, t)

    #     init_c = np.min([init, bc_max])
    #     result = minimize(obj, init_c, bounds=bounds, method=self.par.opt_method, tol=self.par.opt_tol, options={'maxiter':self.par.opt_maxiter})

    #     return result.fun, result.x[0]


    def budget_constraint(self, a, h, s, k, t):
        par = self.par

        if par.retirement_age + par.m <= t:
            return par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + par.chi)
        
        elif par.retirement_age <= t < par.retirement_age + par.m:
            return par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + (1/par.m)*s + par.chi)

        else:
            return par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + (1-par.tau)*h*self.wage(k))


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
                sim.c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol.c[t], sim.a[i,t], sim.s[i,t], sim.k[i,t])
                sim.h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol.h[t], sim.a[i,t], sim.s[i,t], sim.k[i,t])

                # iii. store next-period states
                if t<par.retirement_age:
                    sim.w[i,t] = self.wage(sim.k[i,t])
                    sim.a[i,t+1] = (1.0+par.r_a)*sim.a[i,t] + (1-par.tau)*sim.h[i,t]*sim.w[i,t] - sim.c[i,t]
                    sim.s[i,t+1] = (1+par.r_s)*sim.s[i,t] + par.tau*sim.h[i,t]*sim.w[i,t]
                    sim.k[i,t+1] = (1-par.delta)*sim.k[i,t] + sim.h[i,t]

                elif par.retirement_age <= t < par.retirement_age + par.m: 
                    sim.w[i,t] = self.wage(sim.k[i,t])
                    sim.a[i,t+1] = (1.0+par.r_a)*sim.a[i,t] + 1/par.m*sim.s[i,t] - sim.c[i,t]
                    sim.s[i,t+1] = (1-1/par.m)*sim.s[i,t]
                    sim.k[i,t+1] = (1-par.delta)*sim.k[i,t] 
                
                elif par.retirement_age + par.m <= t < par.T-1:
                    sim.w[i,t] = self.wage(sim.k[i,t])
                    sim.a[i,t+1] = (1.0+par.r_a)*sim.a[i,t] + par.chi - sim.c[i,t]
                    sim.s[i,t+1] = 0
                    sim.k[i,t+1] = (1-par.delta)*sim.k[i,t]
                
                else:
                    sim.w[i,t] = self.wage(sim.k[i,t])
                    pass


