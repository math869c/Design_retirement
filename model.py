import numpy as np
import pandas as pd
from functions_njit import main_solver_loop, main_simulation_loop

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.quadrature import log_normal_gauss_hermite
from bernoulli_distribution import *
from help_functions_non_njit import *


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
        par.beta   = 9.59140783e-01 # 0.995    # Skal kalibreres
        par.sigma  = 1.15618153e+00     # Skal kalibreres
        par.gamma  = 2.26711205e+00       # Skal kalibreres
        par.mu     = 6.85439042e+00    # Skal kalibreres
        par.a_bar  = 0.001
        par.zeta   = 6.87835658e+00  

        par.gamma_1 = 2.62100351e-05

        # Assets
        par.renten= 0.0211947668 
        par.r_s  = par.renten*(1-0.153)
        par.r_a = par.renten*(1-0.42)

        # assets 
        # par.r_a    = 0.010049
        # par.r_s    = 0.016058 # np.mean(np.array(pd.read_csv("Data/mean_matrix.csv")['rente_pension_sum'])[:60])
        
        # wage and human capital
        par.upsilon = 0.0

        par.w_0 =       136.083656
        par.k_0 =        11.140278
        par.beta_1 =         0.0500726898
        par.beta_2 =        -0.000456
        par.delta =         0.027943
        par.k_0_var =         0.049583

        par.full_time_hours = 1924.0

        par.k_scale = 1.
        # Tax system
        par.labor_market_rate            = 0.08           # "am_sats"
        par.employment_deduction_rate    = 0.095          # "beskfradrag_sats"
        par.bottom_tax_rate              = 0.1213          # "bundskat_sats"
        par.top_tax_rate                 = 0.15           # "topskat_sats"
        par.municipal_tax_rate           = 0.2491         # "kommuneskat_sats"
        par.personal_allowance           = 54648          # "personfradrag"
        par.employment_deduction_cap     = 39564          # "beskfradrag_graense"
        par.top_tax_threshold            = 592553         # "topskat_graense"

        # Retirement system 
        par.retirement_age      = 65 - par.start_age # Time when agents enter pension
        par.range               = 5
        par.first_retirement    = 30
        par.last_retirement     = 45

        par.early_benefits_lag = 1

        par.m = 12.0 # Years with retirement payments

        par.tau = np.array(pd.read_csv("Smooth_data/smooth_indbet.csv")['indbetalingsprocent_sum'])

        par.share_lr = 0.55

        par.efterloen = 19194 * 12

        # Means testing retirement payment
        par.chi_base = 10_000
        par.chi_total = 137_520 #=(7198+462)
        par.rho = 0.309
        par.rho_ef = 0.64

        # hire and fire employment
        df_ekso = eksog_prob_simpel(par)[0]
        par.p_e_0 = np.array(df_ekso['to_0'])
        par.p_e_1 = np.array(df_ekso['to_1'])
        par.p_e_2 = np.array(df_ekso['to_2'])
        par.p_efter = 0.6
        par.transition_length = par.T

        # par.initial_ex = pd.read_csv('data/mean_matrix.csv')['extensive_v2_Mean'][0]
        par.initial_ex = par.p_e_1[0]

        # unemployment benefit
        # early_coefficients = pd.read_csv('coefs_early_benefit.csv', header=None).to_numpy()
        # unemployment_coefficients = pd.read_csv("coefs_unemployment_benefit.csv",header=None).to_numpy()
        par.early_benefit = np.array([np.nanmean(pd.read_csv('Data ny def/mean_matrix.csv')['overfor_2'][:30]) if t < 30 else np.nanmean(pd.read_csv('Data ny def/mean_matrix.csv')['overfor_2'][30:]) for t in range(par.T) ])
        coefs = pd.read_csv("coefs_unemployment_benefit.csv",header=None).to_numpy()
        part_1 = np.hstack([np.vstack([np.arange(70)**i for i in range(2)]).T]) @ coefs 
        par.unemployment_benefit = np.array([part_1[t] if t <30  else  part_1[30] for t in range(par.T)]) 


        # life time 
        par.L = 0.9992 # fra regression og data i sas
        par.f = -0.1195 # fra regression og data i sas
        par.x0 = 74.0520 # fra regression og data i sas
        par.pi = np.array([logistic(i,par.L, par.f, par.x0) for i in range(par.T)] )
        par.pi[-1] = 0.0
        par.pi_el = par.pi.copy()
        # par.pi = np.ones_like(par.pi_el)
        
        par.EL = np.where((sp := np.cumprod(par.pi)) > 0, np.cumsum(sp[::-1])[::-1], 0.0)

        with np.errstate(divide='ignore'):
            par.s_lr_deterministic = np.array([
                (par.r_s * (1 + par.r_s)**par.EL[int(r)]) / ((1 + par.r_s)**par.EL[int(r)] - 1)
                for r in range(par.T)
            ])


        # Welfare system
        par.replacement_rate_bf_start = 8
        par.replacement_rate_bf_end = 6
        par.replacement_rate_af_start = 3
        par.start_before = par.retirement_age-par.replacement_rate_bf_start
        par.end_before = par.retirement_age-par.replacement_rate_bf_end
        par.after_retirement = par.retirement_age +par.replacement_rate_af_start

        # State values
        par.unemp = 0
        par.emp = 1
        par.ret = 2

        # Grids
        par.N_a, par.a_sp, par.a_min, par.a_max = 10, 1.5, 0.1, 10_255_346
        par.N_s, par.s_sp, par.s_min, par.s_max = 10, 1.5, 0.0, 6_884_777

        par.N_k, par.k_sp, par.k_min = 15, 1.5, 0
        par.w_max = 1_564_195      
        # par.k_max = (np.log(1_564_195 / par.full_time_hours) - par.beta_2 * np.arange(par.T)**2) / par.beta_1
        par.k_max = np.arange(par.T) + 40        
        

        par.h_min  = 0.2
        par.h_max  = 1.2

        par.c_min  = 1
        par.c_max  = np.inf

        # Shocks
        par.xi      = 0.02
        par.N_xi    = 10
        par.xi_v, par.xi_p = log_normal_gauss_hermite(par.xi, par.N_xi)

        # Simulation
        par.simT = par.T # number of periods
        par.simN = 50000 # number of individuals

    def update_dependent_parameters(self):
        par = self.par

        # Time
        par.T = 100 - par.start_age # time periods

        # # Retirement system
        # par.first_retirement = par.retirement_age - par.range
        par.last_retirement = 55

        # benefits
        # par.early_benefit = np.array([np.nanmean(pd.read_csv('Data ny def/mean_matrix.csv')['overfor_2'][:30]) if t < par.first_retirement else np.nanmean(pd.read_csv('Data ny def/mean_matrix.csv')['overfor_2'][30:]) for t in range(par.T) ])
        # coefs = pd.read_csv("coefs_unemployment_benefit.csv",header=None).to_numpy()
        # part_1 = np.hstack([np.vstack([np.arange(70)**i for i in range(2)]).T]) @ coefs 
        # par.unemployment_benefit = np.array([part_1[t] if t <(30)  else  part_1[30] for t in range(par.T)]) 

        # survival probabilities
        par.pi = np.array([logistic(i,par.L, par.f, par.x0) for i in range(par.T)] )
        par.pi_el = par.pi.copy()

        par.EL = np.where((sp := np.cumprod(par.pi)) > 0, np.cumsum(sp[::-1])[::-1], 0.0)

        # fire and hire employment
        df_ekso = eksog_prob_simpel(par)[0]
        par.p_e_0 = np.array(df_ekso['to_0'])
        par.p_e_1 = np.array(df_ekso['to_1'])
        par.p_e_2 = np.array(df_ekso['to_2'])
        par.initial_ex = 1 - par.p_e_0[0] + par.p_e_2[0]

        par.transition_length = par.T
        par.xi_v, par.xi_p = log_normal_gauss_hermite(par.xi, par.N_xi)

    def allocate(self):
        """ allocate model """
        
        self.update_dependent_parameters()

        par = self.par
        sol = self.sol

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, par.a_sp)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, par.s_sp)
        par.k_grid = np.array([
            nonlinspace(par.k_min, par.k_max[t], par.N_k, par.k_sp) for t in range(par.T)
        ])
        par.e_grid = [0, 1, 2]
        par.efter_grid = [0, 1]


        shape               = (par.T, par.N_a, par.N_s, par.N_k, par.last_retirement + 1, len(par.e_grid), len(par.efter_grid))
        sol.a               = np.full(shape, np.nan)
        sol.ex              = np.full(shape, np.nan)
        sol.c               = np.full(shape, np.nan)
        sol.c_un            = np.full(shape, np.nan)
        sol.h               = np.full(shape, np.nan)
        sol.V               = np.full(shape, np.nan)
        sol.V_employed      = np.full(shape, np.nan)
        sol.V_unemployed    = np.full(shape, np.nan)

        self.allocate_sim()


    def allocate_sim(self):

        self.update_dependent_parameters()

        par = self.par
        sim = self.sim

        np.random.seed(2025)

        shape = (par.simN,par.simT)

        sim.c           = np.nan + np.zeros(shape)
        sim.h           = np.nan + np.zeros(shape)
        sim.a           = np.nan + np.zeros(shape)
        sim.s           = np.nan + np.zeros(shape)
        sim.k           = np.nan + np.zeros(shape)
        sim.e           = np.nan + np.zeros(shape)
        sim.w           = np.nan + np.zeros(shape)
        sim.ex          = np.nan + np.zeros(shape)
        sim.chi_payment = np.nan + np.zeros(shape)
        sim.tax_rate    = np.nan + np.zeros(shape)
        sim.income      = np.nan + np.zeros(shape)
        sim.ret_flag    = np.zeros(shape)
        sim.s_retirement_contrib = np.nan + np.zeros(shape)
        sim.income_before_tax_contrib = np.nan + np.zeros(shape)
        sim.xi          = np.random.choice(par.xi_v, size=(par.simN, par.simT), p=par.xi_p)

        sim.e_state_exogenous = Categorical(p=[par.p_e_0, par.p_e_1, par.p_e_2], size =(par.simN, par.transition_length)).rvs()
        sim.efter_init = Bernoulli(p = par.p_efter, size =(par.simN)).rvs()

        # sim.from_employed   = Categorical(p=[par.p_e_0, par.p_e_1, par.p_e_2], size =(par.simN, par.transition_length)).rvs()
        # sim.from_unemployed = Categorical(p=[par.p_e_0, par.p_e_1, par.p_e_2], size =(par.simN, par.transition_length)).rvs()
        # sim.from_unemployed_to_only_early = Bernoulli(p = par.p_e_2, size =(par.simN, par.transition_length)).rvs()
        # sim.from_employed_to_unemployed = Bernoulli(p = par.p_e_2 + par.p_e_0, size =(par.simN, par.transition_length)).rvs()

        # e. initialization
        sim.a_init, sim.s_init, sim.w_init = [
            np.minimum(initial_value, max_value) for initial_value, max_value in zip(draw_initial_values(par.simN), [par.a_max, par.s_max, par.w_max])
        ]
        sim.k_init                          = np.maximum((np.log(sim.w_init) - np.log(par.w_0))/par.beta_1, 0)
        sim.e_init = Bernoulli(p=par.initial_ex, size=par.simN).rvs()

        sim.s_retirement                    = np.zeros(par.simN)
        sim.retirement_age                  = np.zeros(par.simN)
        sim.retirement_age_idx              = np.zeros(par.simN)
        sim.s_lr_init                       = np.zeros(par.simN)
        sim.s_rp_init                       = np.zeros(par.simN)
        sim.replacement_rate                = np.zeros(par.simN)
        sim.consumption_replacement_rate    = np.zeros(par.simN)
        
    # Solve the model
    def solve(self, do_print = False):
        self.update_dependent_parameters()        
        self.allocate()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            sol.c[:, :, :, :, :, :, :], sol.h[:, :, :, :, :, :, :], sol.ex[:, :, :, :, :, :, :], sol.V[:, :, :, :, :, :, :], sol.a[:, :, :, :, :, :, :] = main_solver_loop(par, sol, do_print)

    def simulate(self):
        self.update_dependent_parameters()        
        self.allocate_sim()

        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim 
            sim.a[:,:], sim.s[:,:], sim.k[:,:], sim.c[:,:], sim.h[:,:], sim.w[:,:], sim.ex[:,:], sim.e[:,:], sim.chi_payment[:,:], sim.tax_rate[:,:], sim.income_before_tax_contrib[:,:], sim.s_retirement[:], sim.retirement_age[:], sim.income[:,:], sim.ret_flag[:,:] = main_simulation_loop(par, sol, sim)

