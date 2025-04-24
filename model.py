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
        par.beta   = 1.0 # 0.995    # Skal kalibreres
        par.sigma  = 1.23151331     # Skal kalibreres
        par.gamma  = 4.738944       # Skal kalibreres
        par.mu     = 0.0 # 7.56180656     # Skal kalibreres
        par.a_bar  = 0.001
        
        # assets 
        par.r_a    = 0.01028688 
        par.r_s    = 0.0147198
        
        # wage and human capital
        par.upsilon = 0.0

        par.k_0 =       154.718555
        par.beta_1 =         0.034528
        par.beta_2 =        -0.000624
        par.delta =         0.001321
        par.k_0_var =         8.465426

        par.full_time_hours = 1924.0

        # Tax system
        par.labor_market_rate            = 0.08           # "am_sats"
        par.employment_deduction_rate    = 0.875          # "beskfradrag_sats"
        par.bottom_tax_rate              = 0.113          # "bundskat_sats"
        par.top_tax_rate                 = 0.15           # "topskat_sats"
        par.municipal_tax_rate           = 0.2491         # "kommuneskat_sats"
        par.personal_allowance           = 46000          # "personfradrag"
        par.employment_deduction_cap     = 33300          # "beskfradrag_graense"
        par.top_tax_threshold            = 498900         # "topskat_graense"

        # Retirement system 
        par.retirement_age      = 65 - par.start_age # Time when agents enter pension
        par.range               = 5
        par.first_retirement    = par.retirement_age - par.range
        par.last_retirement     = par.retirement_age + par.range
        par.retirement_window   = par.last_retirement - par.first_retirement + 1

        par.m = 12 # Years with retirement payments

        par.tau = np.array(pd.read_csv('Data/mean_matrix.csv')["indbetalingsprocent_Mean"].fillna(0))
        par.tau[:] = 0.10


        par.share_lr = 2/3

        # Means testing retirement payment
        par.chi_base = 87_576 # maks beløb, hvorefter ens indkomst trækkes fra 
        par.chi_extra_start = 99_948
        par.chi_max = 95_800
        par.rho = 0.309

        # hire and fire employment
        par.alpha_f0 = 0.043779862783
        par.alpha_f1 = -0.00218450969
        par.alpha_f2 = 0.0000600717239

        par.alpha_h0 = 0.4693704319
        par.alpha_h1 = -0.004887608808
        par.alpha_h2 = 0.000098401435

        par.alpha_e0 = 0.043779862783
        par.alpha_e1 = 0.00218450969
        par.alpha_e2 = 0.0000600717239

        par.transition_length = par.last_retirement + 5
        par.fire = np.minimum(np.maximum(par.alpha_f0 + par.alpha_f1 * np.arange(par.transition_length) + par.alpha_f2 * np.arange(par.transition_length)**2,0),1)
        par.hire = np.minimum(np.maximum(par.alpha_h0 + par.alpha_h1 * np.arange(par.transition_length) + par.alpha_h2 * np.arange(par.transition_length)**2,0),1)
        par.p_early = np.minimum(np.maximum(par.alpha_e0 + par.alpha_e1 * np.arange(par.transition_length) + par.alpha_e2 * np.arange(par.transition_length)**2,0),1)/10

        par.initial_ex = pd.read_csv('data/mean_matrix.csv')['extensive_margin_Mean'][0]

        # unemployment benefit
        par.unemployment_benefit =   70_000
        par.early_benefit =  150_000

        # life time 
        par.L = 0.9992 # fra regression og data i sas
        par.f = -0.1195 # fra regression og data i sas
        par.x0 = 74.0520 # fra regression og data i sas
        par.pi = np.array([logistic(i,par.L, par.f, par.x0) for i in range(par.T)] )
        par.pi[-1] = 0.0
        par.pi_el = par.pi.copy()
        par.pi = np.ones_like(par.pi_el)
        
        par.EL = np.zeros(par.last_retirement + 1)
        for r in range(par.last_retirement + 1):
            par.EL[r] = sum(np.cumprod(par.pi_el[int(r):])*np.arange(int(r),par.T))/(par.T-int(r))

        # Welfare system
        par.replacement_rate_bf_start = 6
        par.replacement_rate_bf_end = 3
        par.replacement_rate_af_start = 5
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

        par.N_k, par.k_sp, par.k_min = 10, 1.5, 50
        par.w_max = 1_564_195      
        par.k_max = (np.log(1_564_195 / par.full_time_hours) - par.beta_2 * np.arange(par.T)**2) / par.beta_1        
        

        par.h_min  = 0.05
        par.h_max  = 1.2

        par.c_min  = 1
        par.c_max  = np.inf

        # Shocks
        par.xi      = 0.01
        par.N_xi    = 10
        par.xi_v, par.xi_p = log_normal_gauss_hermite(par.xi, par.N_xi)

        # Simulation
        par.simT = par.T # number of periods
        par.simN = 10000 # number of individuals

    def update_dependent_parameters(self):
        par = self.par

        # Retirement system
        par.first_retirement = par.retirement_age - par.range
        par.last_retirement = par.retirement_age + par.range
        par.retirement_window = par.last_retirement - par.first_retirement + 1

        # Welfare system
        par.start_before = par.retirement_age - par.replacement_rate_bf_start
        par.end_before = par.retirement_age - par.replacement_rate_bf_end
        par.after_retirement = par.retirement_age + par.replacement_rate_af_start

        # fire and hire employment
        par.transition_length = par.last_retirement + 5
        par.fire = np.minimum(np.maximum(par.alpha_f0 + par.alpha_f1 * np.arange(par.transition_length) + par.alpha_f2 * np.arange(par.transition_length)**2,0),1)
        par.hire = np.minimum(np.maximum(par.alpha_h0 + par.alpha_h1 * np.arange(par.transition_length) + par.alpha_h2 * np.arange(par.transition_length)**2,0),1)
        par.p_early = np.minimum(np.maximum(par.alpha_e0 + par.alpha_e1 * np.arange(par.transition_length) + par.alpha_e2 * np.arange(par.transition_length)**2,0),1)/10


    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, par.a_sp)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, par.s_sp)
        par.k_grid = np.array([
            nonlinspace(par.k_min, par.k_max[t], par.N_k, par.k_sp) for t in range(par.T)
        ])
        par.e_grid = [0, 1, 2]


        shape               = (par.T, par.N_a, par.N_s, par.N_k, par.last_retirement + 1, len(par.e_grid))
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
        sim.s_retirement_contrib = np.nan + np.zeros(shape)
        sim.income_before_tax_contrib = np.nan + np.zeros(shape)
        sim.xi          = np.random.choice(par.xi_v, size=(par.simN, par.simT), p=par.xi_p)

        sim.from_employed   = Categorical(p=[par.fire, 1- par.fire-par.p_early, par.p_early], size =(par.simN, par.transition_length)).rvs()
        sim.from_unemployed = Categorical(p=[1- par.hire-par.p_early, par.hire, par.p_early], size =(par.simN, par.transition_length)).rvs()
        sim.from_unemployed_to_only_early = Bernoulli(p = par.p_early, size =(par.simN, par.transition_length)).rvs()
        sim.from_employed_to_unemployed = Bernoulli(p = par.p_early + par.fire, size =(par.simN, par.transition_length)).rvs()

        # e. initialization
        sim.a_init, sim.s_init, sim.w_init = [
            np.minimum(initial_value, max_value) for initial_value, max_value in zip(draw_initial_values(par.simN), [par.a_max, par.s_max, par.w_max])
        ]
        sim.k_init                          = np.log(sim.w_init)/par.beta_1
        
        # sim.k_init                          = np.clip(np.random.normal(par.k_0, par.k_0_var, par.simN), 0, np.inf)
        # sim.w_init                          = np.exp(np.log(par.w_0) + par.beta_1*sim.k_init)

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
        self.allocate_sim()

        with jit(self) as model:

            par = model.par
            sol = model.sol

            sol.c[:, :, :, :, :, :], sol.h[:, :, :, :, :, :], sol.ex[:, :, :, :, :, :], sol.V[:, :, :, :, :, :], sol.a[:, :, :, :, :, :] = main_solver_loop(par, sol, do_print)

    def simulate(self):
        self.update_dependent_parameters()        
        self.allocate_sim()

        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim 
            sim.a[:,:], sim.s[:,:], sim.k[:,:], sim.c[:,:], sim.h[:,:], sim.w[:,:], sim.ex[:,:], sim.e[:,:], sim.chi_payment[:,:], sim.tax_rate[:,:], sim.income_before_tax_contrib[:,:], sim.s_retirement[:], sim.retirement_age[:], sim.income[:,:] = main_simulation_loop(par, sol, sim)

