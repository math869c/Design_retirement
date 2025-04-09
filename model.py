import numpy as np
import pandas as pd
from functions_njit import main_solver_loop, main_simulation_loop

from EconModel import EconModelClass, jit

from consav.grids import nonlinspace
from consav.quadrature import log_normal_gauss_hermite
from bernoulli_distribution import Bernoulli
from help_functions_non_njit import draw_initial_values


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
        par.upsilon = 0.4

        # par.w_0 =       164.649110
        # par.beta_1 =         0.045522
        # par.beta_2 =        -0.000287
        # par.delta =         0.056238
        # par.k_0_var =         6.482505

        # par.k_0 =             5



        par.k_0 =       154.718555
        par.beta_1 =         0.034528
        par.beta_2 =        -0.000624
        par.delta =         0.001321
        par.k_0_var =         8.465426

        par.full_time_hours = 1924.0

        # Tax system
        par.L1 = 0.3833
        par.L2 = 0.1265
        par.K1 = 6.315E-06
        par.K2 = 1.775E-06
        par.threshold = 551903.0

        # Retirement system 
        par.retirement_age      = 65 - par.start_age # Time when agents enter pension
        par.range               = 5
        par.first_retirement    = par.retirement_age - par.range
        par.last_retirement     = par.retirement_age + par.range
        par.retirement_window   = par.last_retirement - par.first_retirement + 1

        par.m = 12 # Years with retirement payments

        par.tau = np.array(pd.read_csv('Data/mean_matrix.csv')["indbetalingsprocent_Mean"].fillna(0))

        par.share_lr = 2/3

        # Means testing retirement payment
        par.chi_base = 87_576 # maks beløb, hvorefter ens indkomst trækkes fra 
        par.chi_extra_start = 99_948
        par.chi_max = 95_800
        par.rho = 0.309

        # hire and fire employment
        par.alpha_f0 = 0.05
        par.alpha_f1 = -0.01
        par.alpha_f2 = 0.001

        par.alpha_h0 = 0.2
        par.alpha_h1 = 0.05
        par.alpha_h2 = -0.001

        par.fire = np.minimum(np.maximum(par.alpha_f0 + par.alpha_f1 * np.arange(par.T) + par.alpha_f2 * np.arange(par.T)**2,0),1)
        par.hire = np.minimum(np.maximum(par.alpha_h0 + par.alpha_h1 * np.arange(par.T) + par.alpha_h2 * np.arange(par.T)**2,0),1)

        par.initial_ex = pd.read_csv('data/mean_matrix.csv')['extensive_margin_Mean'][0]

        # unemployment benefit
        par.unemployment_benefit = 159_876

        # life time 
        df = pd.read_csv('Data/overlevelses_ssh.csv')
        par.pi_el =  np.array(df[(df['aar'] == 2018) & (df['koen'] == 'Mand') & (df['alder'] <100)].survive_koen_r1)
        par.pi_el[-1] = 0.0
        # par.EL = sum(np.cumprod(par.pi[par.retirement_age:])*np.arange(par.retirement_age,par.T))/(par.T-par.retirement_age) # forventet livstid tilbage efter pension
        par.pi = np.ones_like(par.pi_el)


        # Welfare system
        par.replacement_rate_bf_start = 6
        par.replacement_rate_bf_end = 3
        par.replacement_rate_af_start = 5
        par.start_before = par.retirement_age-par.replacement_rate_bf_start
        par.end_before = par.retirement_age-par.replacement_rate_bf_end
        par.after_retirement = par.retirement_age +par.replacement_rate_af_start

        # Grids

        par.max_a = 10_255_346
        par.N_a, par.a_sp, par.a_min, par.a_max = 10, 1.0, 0.1, par.max_a

        par.max_s = 6_884_777
        par.N_s, par.s_sp, par.s_min, par.s_max = 10, 1.2, 0.0, par.max_s

        par.max_w = 1_564_195
        par.max_k = (1 / par.beta_1) * np.log((par.max_w / par.full_time_hours))
        par.N_k, par.k_sp, par.k_min, par.k_max = 10, 1.1, 0.0, par.max_k

        par.h_min  = 0.19
        par.h_max  = 1.2

        par.c_min  = 1
        par.c_max  = np.inf

        # Shocks
        par.xi      = 0.02
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
        par.fire = np.minimum(np.maximum(par.alpha_f0 + par.alpha_f1 * np.arange(par.T) + par.alpha_f2 * np.arange(par.T)**2,0),1)
        par.hire = np.minimum(np.maximum(par.alpha_h0 + par.alpha_h1 * np.arange(par.T) + par.alpha_h2 * np.arange(par.T)**2,0),1)



    def allocate(self):
        """ allocate model """

        par = self.par
        sol = self.sol

        par.simT = par.T

        par.a_grid = nonlinspace(par.a_min, par.a_max, par.N_a, par.a_sp)
        par.s_grid = nonlinspace(par.s_min, par.s_max, par.N_s, par.s_sp)
        par.k_grid = nonlinspace(par.k_min, par.k_max, par.N_k, par.k_sp)
        par.e_grid = [0, 1]


        shape               = (par.T, par.N_a, par.N_s, par.N_k, par.retirement_window, len(par.e_grid))
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

        sim.e_f         = Bernoulli(p = par.fire, size=shape).rvs()
        sim.e_h         = Bernoulli(p = par.hire, size=shape).rvs()

        # e. initialization
        sim.a_init, sim.s_init, sim.w_init  = draw_initial_values(par.simN)
        sim.k_init                          = np.clip(np.random.normal(par.k_0, par.k_0_var, par.simN), 0, np.inf)
        sim.w_init                          = np.exp(par.beta_1*sim.k_init)

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

            sol.c[:, :, :, :, :, :], sol.c_un[:, :, :, :, :, :], sol.h[:, :, :, :, :, :], sol.ex[:, :, :, :, :, :], sol.V[:, :, :, :, :, :] = main_solver_loop(par, sol, do_print)

    def simulate(self):
        self.update_dependent_parameters()        
        self.allocate_sim()

        with jit(self) as model:

            par = model.par
            sol = model.sol
            sim = model.sim 
            sim.a[:,:], sim.s[:,:], sim.k[:,:], sim.c[:,:], sim.h[:,:], sim.w[:,:], sim.ex[:,:], sim.chi_payment[:,:], sim.tax_rate[:,:], sim.income_before_tax_contrib[:,:], sim.s_retirement[:], sim.retirement_age[:], sim.income[:,:] = main_simulation_loop(par, sol, sim)

