from numba import njit, prange, typeof
import numpy as np 
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from consav.golden_section_search import optimizer

USE_JIT = True  # Set to False to disable JIT for debugging


def jit_if_enabled(parallel=False , fastmath=False):
    """ Apply @njit only if USE_JIT is True """
    return njit(parallel=parallel) if USE_JIT else (lambda f: f)


@jit_if_enabled(parallel=False)
def budget_constraint(par, sol_V, a, h, s, k, t):

    if par.retirement_age + par.m <= t:

        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + par.chi)

    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s
        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + s_retirement/par.m  + par.chi)

    else:
        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + (1-par.tau)*h*wage(par, sol_V, k))

@jit_if_enabled()
def utility(par, sol_V,  c, h):

    return (c)**(1-par.sigma)/(1-par.sigma) - (h)**(1+par.gamma)/(1+par.gamma)

@jit_if_enabled()
def bequest(par, sol_V,  a):

    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled()
def wage(par, sol_V,  k):

    return np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*k**2)

@jit_if_enabled()
def value_next_period_after_reti(par, sol_V,  c, a):
    h = 0.0

    a_next = (1+par.r_a)*a - c

    return utility(par, sol_V, c, h) + bequest(par, sol_V, a_next)

# Value functions med forskellige state inputs
@jit_if_enabled()
def value_function_after_pay(par, sol_V,  c, a, t):

    hours = 0.0
    V_next = sol_V[t+1,:,0,0]
    a_next = (1+par.r_a)*a + par.chi - c
    EV_next = interp_1d(par.a_grid, V_next, a_next)
    
    return utility(par, sol_V, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled()
def value_function_under_pay(par, sol_V,  c, a, s, t):

    hours = 0.0
    V_next = sol_V[t+1,:,:,0]
    s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s # skaleres op for den oprindelige s, naar man gaar på pension.
    a_next = (1+par.r_a)*a + s_retirement/par.m + par.chi - c
    s_next = s-s_retirement/par.m 
    
    EV_next = interp_2d(par.a_grid,par.s_grid, V_next, a_next,s_next)


    return utility(par, sol_V, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled()
def value_function(par, sol_V,  c, h, a, s, k, t):


    V_next = sol_V[t+1]
    
    a_next = (1+par.r_a)*a + (1-par.tau)*h*wage(par, sol_V, k) - c
    s_next = (1+par.r_s)*s + par.tau*h*wage(par, sol_V, k)

    if t < par.retirement_age:
        EV_next = 0.0
        for idx in np.arange(par.N_xi):
            k_next = ((1-par.delta)*k + h)*par.xi_v[idx]
            V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid, V_next, a_next, s_next, k_next)
            EV_next += V_next_interp*par.xi_p[idx]


    return utility(par, sol_V, c, h) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)


@jit_if_enabled()
def obj_consumption(c, par, sol_V, h, a, s, k, t):
    return -value_function(par,sol_V, c, h, a, s, k, t)


@jit_if_enabled()
def obj_hours(h, par, sol_V, a, s, k, t):
    """ 
    1. Given h, find c* that maximizes the value function
    2. Return -V(c*, h)
    """
    # Budget constraint for c given h
    bc_min, bc_max = budget_constraint(par, sol_V, a, h, s, k, t)
    
    # 1D golden-section search over consumption
    c_star = optimizer(
        obj_consumption,     # your negative-value function
        bc_min, 
        bc_max,
        args=(par, sol_V, h, a, s, k, t),
        tol=par.opt_tol
    )
    
    # Return the negative of the maximum value at (h, c_star)
    val_at_c_star = value_function(par, sol_V, c_star, h, a, s, k, t)
    return -val_at_c_star

@jit_if_enabled()
def obj_consumption_after_pay(c, par, sol_V, a, t):
    """ negative of value_function_after_pay(par,sol_V,c,a,t) """
    return -value_function_after_pay(par, sol_V, c, a, t)

@jit_if_enabled()
def obj_consumption_under_pay(c, par, sol_V, a, s, t):
    """ negative of value_function_under_pay(par,sol_V,c,a,s,t) """
    return -value_function_under_pay(par, sol_V, c, a, s, t)


@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol):

    savings_place, human_capital_place, hours_place = 0, 0, 0

    sol_c = sol.c
    sol_h = sol.h
    sol_V = sol.V

    for t in range(par.T - 1, -1, -1):
        print(f"We are in t = {t}")

        for a_idx in prange(len(par.a_grid)):
            assets = par.a_grid[a_idx]

            for s_idx in range(len(par.s_grid)):
                savings = par.s_grid[s_idx]

                for k_idx in range(len(par.k_grid)):
                    human_capital = par.k_grid[k_idx]

                    idx = (t, a_idx, s_idx, k_idx)

                    if t == par.T - 1:
                        # Analytical solution in the last period
                        if par.mu != 0.0:
                            # With bequest motive
                            a_next = (1+par.r_a)*assets+par.chi+par.a_bar - sol_c[idx]

                            sol_c[idx] = (1/(1-par.mu**(-1/par.sigma))) * ((1+par.r_a)*assets+par.chi+par.a_bar) + par.c_bar
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx],a_next)
                        else: 
                            # No bequest motive
                            a_next = par.a_bar
                           
                            sol_c[idx] = (1+par.r_a)*assets+par.chi + par.c_bar
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx],a_next)

                    elif par.retirement_age +par.m <= t:

                        bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, savings_place, human_capital_place, t)
                        
                        c_star = optimizer(
                            obj_consumption_after_pay,
                            bc_min,
                            bc_max,
                            args=(par, sol_V, assets, t),
                            tol=par.opt_tol
                        )

                        sol_c[idx] = c_star
                        sol_h[idx] = hours_place
                        sol_V[idx] = value_function_after_pay(par, sol_V, c_star, assets, t)

                    else:

                        if par.retirement_age <= t:

                            bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, savings, human_capital_place, t)
                            
                            c_star = optimizer(
                                obj_consumption_under_pay,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, assets, savings, t),
                                tol=par.opt_tol
                            )

                            sol_c[idx] = c_star 
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_function_under_pay(par, sol_V, c_star, assets, savings, t)

                        else:

                            idx = (t, a_idx, s_idx, k_idx)

                            h_star = optimizer(
                                obj_hours,         # the hours objective
                                par.h_min,
                                par.h_max,
                                args=(par, sol_V, assets, savings, human_capital, t),
                                tol=par.opt_tol
                            )

                            bc_min, bc_max = budget_constraint(par, sol_V, assets, h_star, savings, human_capital, t)
                            c_star = optimizer(
                                obj_consumption,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, h_star, assets, savings, human_capital, t),
                                tol=par.opt_tol
                            )

                            sol_h[idx] = h_star
                            sol_c[idx] = c_star
                            sol_V[idx] = value_function(par, sol_V, c_star, h_star, assets, savings, human_capital, t)


    return sol_c, sol_h, sol_V