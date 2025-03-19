import numpy as np
import pandas as pd
from functions_njit import main_solver_loop, wage, retirement_payment, utility, bequest, main_simulation_loop

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from optimizers import interp_3d_vec
from consav.quadrature import log_normal_gauss_hermite

from scipy.optimize import root_scalar


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
        par.T = 100 - par.start_age # time periods
               
        # Preferences
        par.beta   = 0.99594784    # Skal kalibreres
        par.sigma  = 1.23151331     # Skal kalibreres
        par.gamma  = 4.738944       # Skal kalibreres
        par.mu     = 7.56180656     # Skal kalibreres
        par.a_bar  = 0.001
        
        # assets 
        par.r_a    = 0.01028688 
        par.r_s    = 0.0147198
        par.H      = 135_000
        df = pd.read_csv("Data/formue_cohort.csv")
        par.s_init = np.array(df[(df['KOEN']==1) & (df['ALDER']==30)]['FGCX'])

        # Unemployment
        par.benefit = 100_000
        
        # wage and human capital
        par.upsilon = 0.4

        par.delta  = 0.056530
        par.beta_1 = 0.034810
        par.beta_2 = -0.000227
        par.w_0             = 208.682235                           
        par.full_time_hours = 1924.0
        par.work_cost       = 1.000          # Skal kalibreres
        
        # Retirement system 
        par.retirement_age = 65 - par.start_age # Time when agents enter pension
        par.m = 12 # Years with retirement payments

        df = pd.read_csv('Data/indbetalinger_koen.csv')
        par.tau = np.concatenate((np.array(df[df['gender'] == "Man"]['indbetalingsprocent']), np.zeros(35)))

        par.chi    = (1-par.upsilon) * np.concatenate((
                        np.zeros(35), 
                        np.array(pd.read_excel("Data/public_pension.xlsx", skiprows=2, index_col=0)["pension"])[:5], 
                        np.tile(np.array(pd.read_excel("Data/public_pension.xlsx", skiprows=2, index_col=0)["pension"])[5], 35)
                    )) 
        par.share_lr = 2/3

        # Means testing retirement payment
        par.chi_base = 87_576 # maks beløb, hvorefter ens indkomst trækkes fra 
        par.chi_extra_start = 99_948
        par.chi_max = 95_800
        par.rho = 0.309

        # life time 
        df = pd.read_csv('Data/overlevelses_ssh.csv')

        par.pi =  np.array(df[(df['aar'] == 2018) & (df['koen'] == 'Mand') & (df['alder'] <100)].survive_koen_r1)
        par.pi[-1] = 0.0
        par.EL = sum(np.cumprod(par.pi[par.retirement_age:])*np.arange(par.retirement_age,par.T))/(par.T-par.retirement_age) # forventet livstid tilbage efter pension

        # welfare measurements 
        par.replacement_rate_bf_start = 10
        par.replacement_rate_bf_end = 7
        par.replacement_rate_af_start = 1

        # Grids
        par.N_a, par.a_sp, par.a_min, par.a_max = 10, 1.0, 0.1, 2_000_000
        par.N_s, par.s_sp, par.s_min, par.s_max = 10, 1.0, 0.0, 2_000_000
        par.N_k, par.k_sp, par.k_min, par.k_max = 10, 1.0, 0.0, 30

        par.h_min  = 0.19
        par.h_max  = 1.2

        par.c_min  = 0.001
        par.c_max  = np.inf

        # Shocks
        par.xi = 0.1
        par.N_xi = 10
        par.xi_v, par.xi_p = log_normal_gauss_hermite(par.xi, par.N_xi)

        # Simulation
        par.simT = par.T # number of periods
        par.simN = 10 # number of individuals


    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, par.a_sp)
        # par.s_grid = [
        #     nonlinspace(par.s_min, par.s_max, par.N_s, par.s_sp) if t < par.retirement_age 
        #     else nonlinspace(par.s_min, 
        #                     par.s_max 
        #                     - (par.s_max/par.m) * (t - par.retirement_age) * (1-par.share_lr)
        #                     - (par.s_max/par.EL) * (t - par.retirement_age) * par.share_lr, 
        #                     par.N_s, par.s_sp) 
        #                     if par.retirement_age <= t < par.retirement_age + par.m
        #     else nonlinspace(par.s_min, 
        #                     par.s_max
        #                     - (par.s_max/par.m) * (par.m) * (1-par.share_lr)
        #                     - (par.s_max/par.EL) * (t - par.retirement_age) * par.share_lr, 
        #                     par.N_s, par.s_sp) 
        #                     if par.retirement_age + par.m <= t < par.retirement_age + par.EL
        #     else -nonlinspace(par.s_min,
        #                     - (par.s_max
        #                     - (par.s_max/par.m) * (par.m) * (1-par.share_lr)
        #                     - (par.s_max/par.EL) * (t - par.retirement_age) * par.share_lr), 
        #                     par.N_s, par.s_sp) 
        #     for t in range(par.T)
        # ]
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, par.s_sp)
        par.k_grid = nonlinspace(par.k_min, par.k_max, par.N_k, par.k_sp)

        shape = (par.T, par.N_a, par.N_s, par.N_k)
        sol.a = np.nan + np.zeros(shape)
        sol.ex = np.nan + np.zeros(shape)
        sol.c = np.nan + np.zeros(shape)
        sol.c_un = np.nan + np.zeros(shape)
        sol.h = np.nan + np.zeros(shape)
        sol.V = np.nan + np.zeros(shape)

        self.allocate_sim()


    def allocate_sim(self):
        par = self.par
        sim = self.sim

        np.random.seed(2025)

        shape = (par.simN,par.simT)

        sim.c = np.nan + np.zeros(shape)
        sim.h = np.nan + np.zeros(shape)
        sim.a = np.nan + np.zeros(shape)
        sim.s = np.nan + np.zeros(shape)
        sim.k = np.nan + np.zeros(shape)
        sim.w = np.nan + np.zeros(shape)
        sim.ex = np.nan + np.zeros(shape)
        sim.chi_payment = np.nan + np.zeros(shape)
        sim.U = np.nan + np.zeros(par.simT)
        sim.EV = np.zeros(1)
        sim.xi = np.random.choice(par.xi_v, size=(par.simN, par.simT), p=par.xi_p)


        # e. initialization
        sim.a_init = np.ones(par.simN)*par.H*np.random.choice(par.xi_v, size=(par.simN), p=par.xi_p)
        sim.s_init = np.ones(par.simN)* par.s_init
        sim.k_init = np.zeros(par.simN)
        sim.w_init = np.ones(par.simN)*par.w_0*np.random.choice(par.xi_v, size=(par.simN), p=par.xi_p)
        sim.s_lr_init = np.zeros(par.simN)
        sim.s_rp_init = np.zeros(par.simN)
        sim.replacement_rate = np.zeros(par.simN)
        sim.consumption_replacement_rate = np.zeros(par.simN)
            


    # Solve the model
    def solve(self, do_print = False):

        with jit(self) as model:

            par = model.par
            sol = model.sol

            sol.c[:, :, :, :], sol.c_un[:, :, :, :], sol.h[:, :, :, :], sol.ex[:, :, :, :], sol.V[:, :, :, :] = main_solver_loop(par, sol, do_print)

    def simulate(self):
        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim 
            sim.a[:,:], sim.s[:,:], sim.k[:,:], sim.c[:,:], sim.h[:,:], sim.w[:,:], sim.ex[:,:], sim.chi_payment[:,:]= main_simulation_loop(par, sol, sim)
          
