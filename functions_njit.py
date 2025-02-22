from numba import njit, prange
import numpy as np 
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from optimizers import optimizer, optimize_outer

from jit_module import jit_if_enabled


@jit_if_enabled(parallel=False)
def budget_constraint(par, sol_V, a, h, s, k, t):

    if par.retirement_age + par.m <= t:

        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + par.chi[t])

    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s
        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + s_retirement/par.m  + par.chi[t])

    else:
        return par.c_min, max(par.c_min*2, (1+par.r_a)*a + (1-par.tau)*h*wage(par, sol_V, k))

@jit_if_enabled(fastmath=True)
def utility(par, sol_V,  c, h):

    return (c)**(1-par.sigma)/(1-par.sigma) - par.work_cost*(h)**(1+par.gamma)/(1+par.gamma)

@jit_if_enabled()
def bequest(par, sol_V,  a):

    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled()
def wage(par, sol_V,  k):

    return par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*k**2)

@jit_if_enabled()
def value_next_period_after_reti(par, sol_V, c, a):
    h = 0.0

    a_next = (1+par.r_a)*a - c

    return utility(par, sol_V, c, h) + bequest(par, sol_V, a_next)

# Value functions med forskellige state inputs
@jit_if_enabled()
def value_function_after_pay(par, sol_V,  c, a, t):

    hours = 0.0
    V_next = sol_V[t+1,:,0,0]
    a_next = (1+par.r_a)*a + par.chi[t] - c
    EV_next = interp_1d(par.a_grid, V_next, a_next)
    
    return utility(par, sol_V, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled()
def value_function_under_pay(par, sol_V,  c, a, s, t):

    hours = 0.0
    V_next = sol_V[t+1,:,:,0]
    s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s # skaleres op for den oprindelige s, naar man gaar på pension.
    a_next = (1+par.r_a)*a + s_retirement/par.m + par.chi[t] - c
    s_next = s-s_retirement/par.m 
    
    EV_next = interp_2d(par.a_grid,par.s_grid, V_next, a_next,s_next)


    return utility(par, sol_V, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled()
def value_function(par, sol_V, sol_EV, c, h, a, s, k, t):

    a_next = (1+par.r_a)*a + (1-par.tau)*h*wage(par, sol_V, k) - c
    s_next = (1+par.r_s)*s + par.tau*h*wage(par, sol_V, k)
    k_next = ((1-par.delta)*k + h/par.h_max)

    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, sol_V, c, h) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)


@jit_if_enabled()
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, t)


@jit_if_enabled()
def obj_hours(h, par, sol_V, sol_EV, a, s, k, t, dist):
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
        args=(par, sol_V, sol_EV, h, a, s, k, t),
        tol=dist
    )
    
    # Return the negative of the maximum value at (h, c_star)
    val_at_c_star = value_function(par, sol_V, sol_EV, c_star, h, a, s, k, t)
    return -val_at_c_star

@jit_if_enabled()
def obj_consumption_after_pay(c, par, sol_V, a, t):
    """ negative of value_function_after_pay(par,sol_V,c,a,t) """
    return -value_function_after_pay(par, sol_V, c, a, t)

@jit_if_enabled()
def obj_consumption_under_pay(c, par, sol_V, a, s, t):
    """ negative of value_function_under_pay(par,sol_V,c,a,s,t) """
    return -value_function_under_pay(par, sol_V, c, a, s, t)

@jit_if_enabled()
def precompute_EV_next(par, sol_V, t):

    V_next = sol_V[t+1]

    EV = np.zeros((len(par.a_grid), len(par.s_grid), len(par.k_grid)))

    for i_a, a_next in enumerate(par.a_grid):
        for i_s, s_next in enumerate(par.s_grid):
            for i_k, k_next in enumerate(par.k_grid):

                EV_val = 0.0
                for idx in range(par.N_xi):
                    k_next = k_next*par.xi_v[idx]  # placeholders for h=0.0
                    V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid, V_next, a_next, s_next, k_next)
                    EV_val += V_next_interp * par.xi_p[idx]

                # Store
                EV[i_a, i_s, i_k] = EV_val

    return EV



@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol):

    savings_place, human_capital_place, hours_place = 0, 0, 0

    sol_c = sol.c
    sol_h = sol.h
    sol_V = sol.V

    for t in range(par.T - 1, -1, -1):
        print(f"We are in t = {t}")

        if t < par.retirement_age:
            sol_EV = precompute_EV_next(par, sol_V, t)

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
                            sol_c[idx] = (1/(1+(par.mu**(1/par.sigma)))) * ((1+par.r_a)*assets+par.chi[t]+par.a_bar) + par.c_bar
                            a_next = (1+par.r_a)*assets+par.chi[t]+par.a_bar - sol_c[idx]

                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx], a_next)

                        else: 
                            # No bequest motive
                            a_next = par.a_bar
                           
                            sol_c[idx] = (1+par.r_a)*assets+par.chi[t] + par.c_bar
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

                            h_star = optimize_outer(
                                obj_hours,         # the hours objective
                                par.h_min,
                                par.h_max,
                                args=(par, sol_V, sol_EV, assets, savings, human_capital, t),
                                tol=par.opt_tol
                            )

                            bc_min, bc_max = budget_constraint(par, sol_V, assets, h_star, savings, human_capital, t)
                            c_star = optimizer(
                                obj_consumption,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                tol=par.opt_tol
                            )

                            sol_h[idx] = h_star
                            sol_c[idx] = c_star
                            sol_V[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)

    return sol_c, sol_h, sol_V