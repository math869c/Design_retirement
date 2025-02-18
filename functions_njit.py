from numba import njit, prange
import numpy as np 
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from consav.golden_section_search import optimizer

USE_JIT = True  # Set to False to disable JIT for debugging


def jit_if_enabled(parallel=False , fastmath=False):
    """ Apply @njit only if USE_JIT is True """
    return njit(parallel=parallel) if USE_JIT else (lambda f: f)


@jit_if_enabled(parallel=False)
def budget_constraint(par, sol, a, h, s, k, t):

    if par.retirement_age + par.m <= t:

        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + par.chi)

    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s
        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + s_retirement/par.m  + par.chi)

    else:
        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + (1-par.tau)*h*wage(par, sol, k))

@jit_if_enabled()
def utility(par, sol,  c, h):

    return (c)**(1-par.sigma)/(1-par.sigma) - (h)**(1+par.gamma)/(1+par.gamma)

@jit_if_enabled()
def bequest(par, sol,  a):

    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled()
def wage(par, sol,  k):

    return np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*k**2)

@jit_if_enabled()
def value_next_period_after_reti(par, sol,  c, a):
    h = 0.0

    a_next = (1+par.r_a)*a - c

    return utility(par, sol, c, h) + bequest(par, sol, a_next)

# Value functions med forskellige state inputs
@jit_if_enabled()
def value_function_after_pay(par, sol,  c, a, t):

    hours = 0.0
    V_next = sol.V[t+1,:,0,0]
    a_next = (1+par.r_a)*a + par.chi - c
    EV_next = interp_1d(par.a_grid, V_next, a_next)
    
    return utility(par, sol, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol, a_next)

@jit_if_enabled()
def value_function_under_pay(par, sol,  c, a, s, t):

    hours = 0.0
    V_next = sol.V[t+1,:,:,0]
    s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s # skaleres op for den oprindelige s, naar man gaar på pension.
    a_next = (1+par.r_a)*a + s_retirement/par.m + par.chi - c
    s_next = s-s_retirement/par.m 
    
    EV_next = interp_2d(par.a_grid,par.s_grid, V_next, a_next,s_next)


    return utility(par, sol, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol, a_next)

@jit_if_enabled()
def value_function(par, sol,  c, h, a, s, k, t):


    V_next = sol.V[t+1]
    
    a_next = (1+par.r_a)*a + (1-par.tau)*h*wage(par, sol, k) - c
    s_next = (1+par.r_s)*s + par.tau*h*wage(par, sol, k)

    if t < par.retirement_age:
        EV_next = 0.0
        for idx in np.arange(par.N_xi):
            k_next = ((1-par.delta)*k + h)*par.xi_v[idx]
            V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid, V_next, a_next, s_next, k_next)
            EV_next += V_next_interp*par.xi_p[idx]


    return utility(par, sol, c, h) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol, a_next)


@jit_if_enabled()
def obj_consumption(c, par, sol, h, a, s, k, t):
    return -value_function(par,sol, c, h, a, s, k, t)


@jit_if_enabled()
def obj_hours(h, par, sol, a, s, k, t):
    """ 
    1. Given h, find c* that maximizes the value function
    2. Return -V(c*, h)
    """
    # Budget constraint for c given h
    bc_min, bc_max = budget_constraint(par, sol, a, h, s, k, t)
    
    # 1D golden-section search over consumption
    c_star = optimizer(
        obj_consumption,     # your negative-value function
        bc_min, 
        bc_max,
        args=(par, sol, h, a, s, k, t),
        tol=par.opt_tol
    )
    
    # Return the negative of the maximum value at (h, c_star)
    val_at_c_star = value_function(par, sol, c_star, h, a, s, k, t)
    return -val_at_c_star

@jit_if_enabled()
def obj_consumption_after_pay(c, par, sol, a, t):
    """ negative of value_function_after_pay(par,sol,c,a,t) """
    return -value_function_after_pay(par, sol, c, a, t)

@jit_if_enabled()
def obj_consumption_under_pay(c, par, sol, a, s, t):
    """ negative of value_function_under_pay(par,sol,c,a,s,t) """
    return -value_function_under_pay(par, sol, c, a, s, t)


@jit_if_enabled(parallel=False)
def main_solver_loop(par, sol):

    idx_s_place, idx_k_place = 0, 0
    savings_place, human_capital_place, hours_place = 0, 0, 0

    for t in range(par.T - 1, -1, -1):
        print(f"We are in t = {t}")

        for a_idx in prange(len(par.a_grid)):
            assets = par.a_grid[a_idx]

            idx = (t, a_idx, np.newaxis, np.newaxis)
            idx_next = (t+1, a_idx, idx_s_place, idx_k_place)

            if t == par.T - 1:
                # Analytical solution in the last period
                if par.mu != 0.0:
                    # With bequest motive
                    a_next = (1+par.r_a)*assets+par.chi+par.a_bar -sol.c[idx]

                    sol.c[idx] = (1/(1-par.mu**(-1/par.sigma))) * ((1+par.r_a)*assets+par.chi+par.a_bar) + par.c_bar
                    sol.h[idx] = hours_place
                    sol.V[idx] = value_next_period_after_reti(par, sol, sol.c[idx],a_next)
                else: 
                    # No bequest motive
                    a_next = par.a_bar
                    
                    sol.c[idx] = (1+par.r_a)*assets+par.chi + par.c_bar
                    sol.h[idx] = hours_place
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

        print(sol.c)

    return sol.c, sol.h, sol.V