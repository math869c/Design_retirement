import numpy as np
import pandas as pd
from functions_njit import main_solver_loop, wage

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from consav.quadrature import log_normal_gauss_hermite


class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """
        pass 

    def setup(self):
        """ set baseline parameters """

        # Unpack
        par = self.par

        # Optimization settings
        par.opt_tol = 1e-6
        par.opt_maxiter = 1000


        # Time
        par.start_age = 30  # Time when agents enter the workforce
        par.retirement_age = 65 - par.start_age # Time when agents enter pension
        par.T = 100 - par.start_age # time periods
        par.m = 10 # Years with retirement payments

        par.scale_hour = 1924

        # Preferences
        par.beta   = 0.98    # Skal kalibreres
        par.sigma  = 1.9     # Skal kalibreres
        par.gamma  = 2.5     # Skal kalibreres
        par.mu     = 0.2     # Skal kalibreres
        par.a_bar  = 1.0
        par.c_bar  = 0.001

        par.r_a    = 0.02
        par.r_s    = 0.04
        par.H      = 135_000

        par.tau    = 0.15
        par.chi    = np.concatenate((
                        np.zeros(35), 
                        np.array(pd.read_excel("Data/public_pension.xlsx", skiprows=2, index_col=0)["pension"])[:5], 
                        np.tile(np.array(pd.read_excel("Data/public_pension.xlsx", skiprows=2, index_col=0)["pension"])[5], 35)
                    )) 
        
        par.delta  = 0.016804

        par.beta_1 = 0.030682
        par.beta_2 = -0.001128

        par.w_0             = 215.033648                           
        par.full_time_hours = 1924
        par.work_cost       = par.w_0*par.full_time_hours          # Skal kalibreres

        df = pd.read_csv('Data\overlevelses_ssh.csv')
        par.pi = 1 - np.array(df[(df['aar'] == 2018) & (df['koen'] == 'Mand') & (df['alder'] <100)].survive_r1)
        par.pi[-1] = 1.0

        # Grids
        par.a_max  = 2_000_000 
        par.a_min  = 0
        par.N_a    = 20
        par.a_sp   = 1

        par.s_max  = 2_000_000
        par.s_min  = 0
        par.N_s    = 20
        par.s_sp   = 1

        par.k_min  = 0
        par.k_max  = 30
        par.N_k    = 20
        par.k_sp   = 1

        par.h_min  = 0
        par.h_max  = 1.2

        par.c_min  = 0.001
        par.c_max  = np.inf

        # Shocks
        par.xi = 0.1
        par.N_xi = 1
        par.xi_v, par.xi_p = log_normal_gauss_hermite(par.xi, par.N_xi)

        # Simulation
        par.simT = par.T # number of periods
        par.simN = 1 # number of individuals


    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, par.a_sp)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, par.s_sp)
        par.k_grid = nonlinspace(par.k_min, par.k_max, par.N_k, par.k_sp)

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

            sol.c[:, :, :, :], sol.h[:, :, :, :], sol.V[:, :, :, :] = main_solver_loop(par, sol)




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
                        sim.a[i,t+1] = (1+par.r_a)*sim.a[i,t] + sim.s_payment[i] + par.chi[t] - sim.c[i,t]
                        sim.s[i,t+1] = sim.s[i,t] - sim.s_payment[i]
                        sim.k[i,t+1] = ((1-par.delta)*sim.k[i,t])*sim.xi[i,t]
                    
                    elif par.retirement_age + par.m <= t < par.T-1:
                        sim.w[i,t] = wage(par, sol, sim.k[i,t])
                        sim.a[i,t+1] = (1+par.r_a)*sim.a[i,t] + par.chi[t] - sim.c[i,t]
                        sim.s[i,t+1] = 0
                        sim.k[i,t+1] = ((1-par.delta)*sim.k[i,t])*sim.xi[i,t]
                    
                    else:
                        sim.w[i,t] = wage(par, sol, sim.k[i,t])
                        pass

