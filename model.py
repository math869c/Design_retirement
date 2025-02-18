import numpy as np
import pandas as pd
from scipy.optimize import minimize
from functions_njit import *


from EconModel import EconModelClass, jit

from numba import njit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from consav.quadrature import log_normal_gauss_hermite
from consav.golden_section_search import optimizer

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
        par.retirement_age = 65 - par.start_age # Time when agents enter pension
        par.T = 100 - par.start_age # time periods
        par.m = 10 # Years with retirement payments

        # Preferences
        par.beta   = 0.90    # discount factor
        par.sigma  = 3.0     # CRRA
        par.gamma  = 2.5    # labor disutility curvature
        par.mu     = 0.0
        par.a_bar  = 1.0
        par.c_bar  = 0.001

        par.r_a    = 0.02
        par.r_s    = 0.04
        par.H      = 0.0
 
        par.tau    = 0.0    # 10% pension contribution
        par.chi    = 0.0     # public pension replacement
        par.delta  = 0.07    # human capital depreciation

        par.beta_1 = 0.01
        par.beta_2 = 0.001    # or a small positive number

        par.w_0    = 30.0

        ages       = np.arange(par.start_age, par.T + par.start_age + 1)
        par.pi     = 1 - np.concatenate((np.ones(8), 
                                     np.array(pd.read_excel('overlevelsesssh.xlsx',sheet_name='Sheet1', engine="openpyxl")['Mand_LVU'])[:-5]/100,
                                     np.zeros(1)))
        # par.pi     = np.zeros((par.T))


        # Grids
        par.a_max  = 150
        par.a_min  = 0
        par.N_a    = 10
        par.a_sp   = 2

        par.s_max  = 10
        par.s_min  = 0
        par.N_s    = 10
        par.s_sp   = 1

        par.k_min  = 0
        par.k_max  = 10
        par.N_k    = 10
        par.k_sp   = 1

        par.h_min  = 0
        par.h_max  = 1

        par.c_min  = 0.001
        par.c_max  = np.inf

        # Shocks
        par.xi = 0.1
        par.N_xi = 10
        par.xi_v, par.xi_p = log_normal_gauss_hermite(par.xi, par.N_xi)

        # Simulation
        par.simT = par.T # number of periods
        par.simN = 1 # number of individuals
        par.simN = 1 # number of individuals






    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, 2)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, 2)
        par.k_grid = nonlinspace(par.k_min, par.k_max, par.N_k, 1)

        shape = (par.T, par.N_a, par.N_s, par.N_k)
        sol.c = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        self.allocate_sim()


    def allocate_sim(self):
        par = self.par
        sim = self.sim

        shape = (par.simN,par.simT)

        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.s = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.w = np.nan + np.zeros(shape)
        sim.xi = np.random.choice(par.xi_v, size=(par.simN, par.simT), p=par.xi_p)


        # e. initialization
        sim.a_init = np.ones(par.simN)*par.H
        sim.s_init = np.zeros(par.simN)
        sim.k_init = np.zeros(par.simN)
        sim.w_init = np.ones(par.simN)*par.w_0
        sim.s_payment = np.zeros(par.simN)



    def solve(self):

        with jit(self) as model:

            par = model.par
            sol = model.sol

            idx_s_place, idx_k_place = 0, 0
            savings_place, human_capital_place, hours_place = 0, 0, 0

            for t in reversed(range(par.T)):
                print(f"We are in t = {t}")


                for a_idx, assets in enumerate(par.a_grid):

                    idx = (t, a_idx, np.newaxis, np.newaxis)
                    idx_next = (t+1, a_idx, idx_s_place, idx_k_place)

                    if t == par.T - 1:
                        # Analytical solution in the last period
                        if par.mu != 0.0:
                            # With bequest motive
                            sol.c[idx] = (1/(1-par.mu**(-1/par.sigma))) * ((1+par.r_a)*assets+par.chi+par.a_bar) + par.c_bar
                            a_next = (1+par.r_a)*assets+par.chi+par.a_bar -sol.c[a_idx,np.newaxis,np.newaxis]
                        else: 
                            # No bequest motive
                            sol.c[idx] = (1+par.r_a)*assets+par.chi + par.c_bar
                            a_next = par.a_bar
                        sol.h[idx] = 0
                        sol.V[idx] = value_next_period_after_reti(par, sol, sol.c[idx],a_next)

                    
                    elif par.retirement_age +par.m <= t:

                        bc_min, bc_max = budget_constraint(par, sol, assets, hours_place, savings_place, human_capital_place, t)
                        
                        c_star = optimizer(
                            obj_consumption_after_pay,
                            bc_min,
                            bc_max,
                            args=(par, sol, assets, t),
                            tol=par.opt_tol
                        )

                        sol.c[idx] = c_star
                        sol.h[idx] = hours_place
                        sol.V[idx] = value_function_after_pay(par, sol, c_star, assets, t)

                    else:
                        for s_idx, savings in enumerate(par.s_grid):
                            idx = (t, a_idx, s_idx, np.newaxis)
                            idx_next = (t+1, a_idx, s_idx, idx_k_place)

                            if par.retirement_age <= t:

                                bc_min, bc_max = budget_constraint(par, sol, assets, hours_place, savings, human_capital_place, t)
                                
                                c_star = optimizer(
                                    obj_consumption_under_pay,
                                    bc_min,
                                    bc_max,
                                    args=(par, sol, assets, savings, t),
                                    tol=par.opt_tol
                                )

                                sol.c[idx] = c_star 
                                sol.h[idx] = hours_place
                                sol.V[idx] = value_function_under_pay(par, sol, c_star, assets, savings, t)

                            else:
                                for k_idx, human_capital in enumerate(par.k_grid):
                                    idx = (t, a_idx, s_idx, k_idx)
                                    idx_next = (t+1, a_idx, s_idx, k_idx)

                                    init_c = sol.c[idx_next]
                                    init_h = sol.h[idx_next]      

                                    h_star = optimizer(
                                        obj_hours,         # the hours objective
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol, assets, savings, human_capital, t),
                                        tol=par.opt_tol
                                    )

                                    bc_min, bc_max = budget_constraint(par, sol, assets, h_star, savings, human_capital, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol, h_star, assets, savings, human_capital, t),
                                        tol=par.opt_tol
                                    )

                                    sol.h[idx] = h_star
                                    sol.c[idx] = c_star
                                    sol.V[idx] = value_function(par, sol, c_star, h_star, assets, savings, human_capital, t)

    def simulate(self):



        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim

        
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
                    if t == par.retirement_age:
                        sim.s_payment[i] = sim.s[i,t]/par.m

                    # iii. store next-period states
                    if t < par.retirement_age:
                        sim.w[i,t] = wage(par, sol, sim.k[i,t])
                        sim.a[i,t+1] = (1+par.r_a)*sim.a[i,t] + (1-par.tau)*sim.h[i,t]*sim.w[i,t] - sim.c[i,t]
                        sim.s[i,t+1] = (1+par.r_s)*sim.s[i,t] + par.tau*sim.h[i,t]*sim.w[i,t]
                        sim.k[i,t+1] = ((1-par.delta)*sim.k[i,t] + sim.h[i,t])*sim.xi[i,t]

                    elif par.retirement_age <= t < par.retirement_age + par.m: 
                        sim.w[i,t] = wage(par, sol, sim.k[i,t])
                        sim.a[i,t+1] = (1+par.r_a)*sim.a[i,t] + sim.s_payment[i] + par.chi - sim.c[i,t]
                        sim.s[i,t+1] = sim.s[i,t] - sim.s_payment[i]
                        sim.k[i,t+1] = ((1-par.delta)*sim.k[i,t])*sim.xi[i,t]
                    
                    elif par.retirement_age + par.m <= t < par.T-1:
                        sim.w[i,t] = wage(par, sol, sim.k[i,t])
                        sim.a[i,t+1] = (1+par.r_a)*sim.a[i,t] + par.chi - sim.c[i,t]
                        sim.s[i,t+1] = 0
                        sim.k[i,t+1] = ((1-par.delta)*sim.k[i,t])*sim.xi[i,t]
                    
                    else:
                        sim.w[i,t] = wage(par, sol, sim.k[i,t])
                        pass

