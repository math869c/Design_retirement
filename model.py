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
        par.opt_tol = 1e-6
        par.opt_maxiter = 1000


        # Time
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


        with jit(self) as model_jit:

            model_jit_par = model_jit.par
            model_jit_sol = model_jit.sol

            for t in reversed(range(par.T)):
                print(f"We are in t = {t}")
                par.stop_parameter = 0

                for a_idx, assets in enumerate(par.a_grid):
                    for s_idx, savings in enumerate(par.s_grid):
                        for k_idx, human_capital in enumerate(par.k_grid):

                            idx = (t, a_idx, s_idx, k_idx)

                            if t == par.T - 1:
                                hours = 0

                                bc_min, bc_max = budget_constraint(model_jit_par, assets, hours, savings, human_capital, t)
                                
                                init_c = par.c_min

                                f_obj = create_obj_last(model_jit_par, assets)
                                optimal_c = golden_section_search(f_obj, bc_min, bc_max, x0=init_c, tol=par.opt_tol, max_iter=par.opt_maxiter)

                                sol.c[idx] = optimal_c
                                sol.h[idx] = hours
                                sol.V[idx] = value_last_period(model_jit_par, optimal_c, assets)

                            elif par.retirement_age <= t:
                                hours = 0

                                bc_min, bc_max = budget_constraint(model_jit_par, assets, hours, savings, human_capital, t)

                                init_c = np.min([optimal_c, bc_max])

                                f_obj = create_obj_c(model_jit_par, model_jit_sol, hours, assets, savings, human_capital, t)
                                optimal_c = golden_section_search(f_obj, bc_min, bc_max, x0=init_c, tol=par.opt_tol, max_iter=par.opt_maxiter)

                                sol.c[idx] = optimal_c
                                sol.h[idx] = hours
                                sol.V[idx] = value_function(model_jit_par, model_jit_sol, optimal_c, hours, assets, savings, human_capital, t)

                            else:
                                init_c = sol.c[(t+1, a_idx, s_idx, k_idx)]

                                if par.retirement_age - 1 == t:
                                    init_h = par.h_max
                                    init_c = par.c_min
                                else:
                                    init_h = optimal_c

                                f_obj = create_obj_hour(model_jit_par, model_jit_sol, assets, savings, human_capital, init_c, t)

                                optimal_h = golden_section_search(f_obj, par.h_min, par.h_max, x0=init_h, tol=par.opt_tol, max_iter=par.opt_maxiter)

                                optimal_c = optimize_consumption(model_jit_par, model_jit_sol, optimal_h, assets, savings, human_capital, init_c, t)

                                sol.c[idx] = optimal_c
                                sol.h[idx] = optimal_h
                                sol.V[idx] = value_function(model_jit_par, model_jit_sol, optimal_c, optimal_h, assets, savings, human_capital, t)





# @njit
def obj_last(x, par, assets):
    return -value_last_period(par, x, assets)

# @njit
def create_obj_last(par, assets):
    # @njit
    def f(x):
        return obj_last(x, par, assets)
    return f



# @njit
def obj_c(x, par, sol, hours, assets, savings, human_capital, t):
    return -value_function(par, sol, x, hours, assets, savings, human_capital, t)

# @njit
def create_obj_c(par, sol, hours, assets, savings, human_capital, t):
    # @njit
    def f(x):
        return obj_c(x, par, sol, hours, assets, savings, human_capital, t)
    return f


# @njit
def obj_hour(x, par, sol, assets, savings, human_capital, init_c, t):
    return optimize_consumption(par, sol, x, assets, savings, human_capital, init_c, t)

# @njit
def create_obj_hour(par, sol, assets, savings, human_capital, init_c, t):
    # @njit
    def f(x):
        return obj_hour(x, par, sol, assets, savings, human_capital, init_c, t)
    return f




# @njit
def optimize_consumption(par, sol, h, a, s, k, init, t):

    bc_min, bc_max = budget_constraint(par, a, h, s, k, t)

    f_obj = create_obj_c(par, sol, h, a, s, k, t)   

    init_c = np.min([init, bc_max])
    optimal_c = golden_section_search(f_obj, par.h_min, par.h_max, x0=init_c, tol=par.opt_tol, max_iter=par.opt_maxiter)

    return optimal_c

# @njit
def budget_constraint(par, a, h, s, k, t):

    if par.retirement_age + par.m <= t:
        return par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + par.chi)
    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        return par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + (1/par.m)*s + par.chi)

    else:
        return par.c_min, max(par.c_min*2, (1.0+par.r_a)*a + (1-par.tau)*h*wage(par, k))

# @njit
def wage(par, k):
    
    return np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*k**2)

# @njit
def value_function(par, sol, c, h, a, s, k, t):

    V_next = sol.V[t+1]
    
    if par.retirement_age + par.m <= t:
        a_next = (1.0+par.r_a)*a + par.chi - c
        s_next = 0
    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        a_next = (1.0+par.r_a)*a + (1/par.m)*s + par.chi - c
        s_next = (1-1/par.m)*s

    else:
        a_next = (1.0+par.r_a)*a + (1-par.tau)*h*wage(par, k) - c
        s_next = (1+par.r_s)*s + par.tau*h*wage(par, k)

    k_next = (1-par.delta)*k + h

    V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid, V_next, a_next, s_next, k_next)

    return utility(par.sigma, par.gamma, c, h) + (1-par.pi[t+1])*par.beta*V_next_interp + par.pi[t+1]*bequest(par.mu, par.a_bar, par.sigma, a_next)

# @njit
def bequest(mu, a_bar, sigma, a):

    return mu*(a+a_bar)**(1-sigma) / (1-sigma)

# @njit
def value_last_period(par, c, a):
    h = 0

    a_next = (1+par.r_a)*a - c

    return utility(par.sigma, par.gamma, c, h) + bequest(par.mu, par.a_bar, par.sigma, a_next)

@njit
def utility(sigma, gamma, c, h):
    return (c)**(1-sigma)/(1-sigma) - (h)**(1+gamma)/(1+gamma)

# @njit
def golden_section_search(f, a, b, x0=None, tol=1e-8, max_iter=1000):
    """
    1D golden-section search to minimize f(x) over [a,b],
    optionally taking an initial guess x0 in [a,b].
    """
    phi = 0.5 * (3.0 - np.sqrt(5.0))  # ~0.618
    # If no initial guess is given, proceed with standard initialization:
    if x0 is None:
        c = a + phi * (b - a)
        d = b - phi * (b - a)
    else:
        # Clamp x0 to [a,b]
        x0 = max(a, min(b, x0))

        # Place c and d around x0 in a 'golden' way:
        c = x0 - phi * (x0 - a)
        d = x0 + phi * (b - x0)

        # Keep c and d within [a,b]:
        if c < a:
            c = a
        if d > b:
            d = b
        # If we ended up with c >= x0 or d <= x0, revert to standard approach:
        if c >= x0 or d <= x0:
            c = a + phi * (b - a)
            d = b - phi * (b - a)

    fc = f(c)
    fd = f(d)

    for _ in range(max_iter):
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = a + phi * (b - a)
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            d = b - phi * (b - a)
            fd = f(d)

        if abs(b - a) < tol:
            break

    # Return midpoint of final bracket
    return 0.5 * (a + b)
