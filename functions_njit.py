from numba import njit, prange
import numpy as np 
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from optimizers import optimizer, optimize_outer, interp_3d_vec

from jit_module import jit_if_enabled

#######################################################################
#######################################################################
#######################################################################
# Structure 
#   1. Essentiel functions for such as utility, bequest, and wage
#   2. Value functions
#   3. helping functions in solving and optimizing
#   4. Objective functions 
#   5. Solving the model
#   6. simulate the model
#######################################################################
#######################################################################
#######################################################################

# 1. Essentiel functions for such as utility, bequest, and wage 
@jit_if_enabled(fastmath=True)
def utility(par, c, h):

    return (c**(1-par.sigma))/(1-par.sigma) - par.work_cost*(h**(1+par.gamma))/(1+par.gamma)

@jit_if_enabled(fastmath=True)
def bequest(par, a):

    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled(fastmath=True)
def wage(par, k, t):

    return (1-par.upsilon)*par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)

@jit_if_enabled(fastmath=True)
def retirement_payment(par, sol_V, a, s, s_lr, t):
    base_payment = par.chi_base

    if par.retirement_age +par.m< t:
        income = s_lr
    else: 
        s_retirement = s / (1-(t-par.retirement_age)*(par.share_lr*(1/par.EL)+(1-par.share_lr)*(1/par.m)))
        s_lr = par.share_lr * (s_retirement/par.EL)
        s_rp = (1-par.share_lr) * (s_retirement/par.m)
        income = s_lr + s_rp

    exceed = np.maximum(0, income - par.chi_max)
    extra_pension = np.maximum(0, par.chi_extra_start - exceed*par.reduction_rate)

    return (1-par.upsilon)*(base_payment + extra_pension)


# 2. Value functions
@jit_if_enabled(fastmath=True)
def value_function_after_pay(par, sol_V,  c, a, s_lr, chi, t):

    hours = 0.0
    V_next = sol_V[t+1,:,0,0]
    a_next = (1+par.r_a)*(a + chi + s_lr - c)
    EV_next = interp_1d(par.a_grid, V_next, a_next)
    
    return utility(par, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, a_next)

@jit_if_enabled(fastmath=True)
def value_function_under_pay(par, sol_V,  c, a, s, chi, t):

    hours = 0.0
    V_next = sol_V[t+1,:,:,0]
    s_retirement = s / (1-(t-par.retirement_age)*(par.share_lr*(1/par.EL)+(1-par.share_lr)*(1/par.m)))
    s_lr = par.share_lr * (s_retirement/par.EL)
    s_rp = (1-par.share_lr) * (s_retirement/par.m)
    # s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s # skaleres op for den oprindelige s, naar man gaar på pension.
    a_next = (1+par.r_a)*(a + s_lr + s_rp + chi - c)
    s_next = s - s_lr - s_rp
    
    EV_next = interp_2d(par.a_grid,par.s_grid, V_next, a_next,s_next)


    return utility(par, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, a_next)

@jit_if_enabled(fastmath=True)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, t):

    a_next = (1+par.r_a)*(a + (1-par.tau[t])*h*wage(par, k, t) - c)
    s_next = (1+par.r_s)*(s + par.tau[t]*h*wage(par, k, t))
    k_next = ((1-par.delta)*k + h)

    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, c, h) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, a_next)

@jit_if_enabled(fastmath=True)
def value_next_period_after_reti(par, sol_V, c, a, chi, t):
    h = 0.0

    a_next = (1+par.r_a)*(a + chi - c)
    
    return utility(par, c, h) + bequest(par, a_next)

# 3. Helping functions in solving and optimizing
@jit_if_enabled(fastmath=True)
def budget_constraint(par, sol_V, a, h, s, k, s_lr, chi, t):

    if par.retirement_age + par.m <= t:
        return par.c_min, max(par.c_min*2, a + chi + s_lr)
    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        s_retirement = s / (1-(t-par.retirement_age)*((par.share_lr)*(1/par.EL)+(1-par.share_lr)*(1/par.m)))
        s_lr = par.share_lr * (s_retirement/par.EL)
        s_rp = (1-par.share_lr) * (s_retirement/par.m)
        return par.c_min, max(par.c_min*2, a + s_lr + s_rp  + chi)

    else:
        return par.c_min, max(par.c_min*2, a + (1-par.tau[t])*h*wage(par, k, t))

@jit_if_enabled(fastmath=True)
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

# 4. Objective functions 
@jit_if_enabled(fastmath=True)
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, t)


@jit_if_enabled(fastmath=True)
def obj_hours(h, par, sol_V, sol_EV, a, s, k, s_lr, chi, t, dist):
    """ 
    1. Given h, find c* that maximizes the value function
    2. Return -V(c*, h)
    """
    # Budget constraint for c given h
    bc_min, bc_max = budget_constraint(par, sol_V, a, h, s, k, s_lr, chi, t)
    
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
def obj_consumption_last_period(c, par, sol_V, a, chi, t):
    """ negative of value_function_after_pay(par,sol_V,c,a,t) """
    return -value_next_period_after_reti(par, sol_V, c, a, chi, t)

@jit_if_enabled()
def obj_consumption_after_pay(c, par, sol_V, a, s_lr, chi, t):
    """ negative of value_function_after_pay(par,sol_V,c,a,t) """
    return -value_function_after_pay(par, sol_V, c, a, s_lr, chi, t)

@jit_if_enabled()
def obj_consumption_under_pay(c, par, sol_V, a, s, chi, t):
    """ negative of value_function_under_pay(par,sol_V,c,a,s,t) """
    return -value_function_under_pay(par, sol_V, c, a, s, chi, t)



# 5. Solving the model
@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol, do_print = False):

    savings_place, human_capital_place, hours_place = 0, 0, 0

    sol_a = sol.a
    sol_c = sol.c
    sol_h = sol.h
    sol_V = sol.V
    
    for t in range(par.T - 1, -1, -1):
        if do_print:
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
                            s_retirement = savings / (1 - ((t-par.retirement_age)/par.EL)*(par.share_lr)-(1-par.share_lr))
                            s_lr = par.share_lr * (s_retirement/par.EL) 
                            chi = retirement_payment(par, sol_V, assets, savings, s_lr, t)

                            sol_c[idx] = ((1/(1+(par.mu*(1+par.r_a))**(-1/par.sigma)*(1+par.r_a)))
                                          * (par.mu*(1+par.r_a))**(-1/par.sigma) 
                                          * ((1+par.r_a)*(assets+chi+s_lr)+par.a_bar))
                            sol_a[idx] = assets + par.a_bar + s_lr + chi - sol_c[idx]

                            # bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, savings, human_capital_place, t)

                            # c_star = optimizer(
                            #     obj_consumption_last_period,
                            #     bc_min,
                            #     bc_max,
                            #     args=(par, sol_V, assets, chi, t),
                            #     tol=par.opt_tol
                            # )

                            # sol_c[idx] = c_star
                            # sol_a[idx] = assets + par.chi[t] + par.a_bar - sol_c[idx]
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx], assets, chi, t)

                        else: 
                            # No bequest motive                          
                            s_retirement = savings / (1 - ((t-par.retirement_age)/par.EL)*par.share_lr-(1-par.share_lr))
                            s_lr = par.share_lr * (s_retirement/par.EL)
                            chi = retirement_payment(par, sol_V, assets, savings, s_lr, t)
                            sol_c[idx] = (1+par.r_a)*(assets+chi + s_lr)
                            sol_a[idx] = assets + chi + par.a_bar - sol_c[idx]
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx], assets, chi, t)

                    elif par.retirement_age +par.m <= t:
                        chi = retirement_payment(par, sol_V, assets, savings, s_lr, t)
                        bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, savings, human_capital_place, s_lr, chi, t) #er s_lr ok her i stedet for svaings? Det virker, bare lidt grimt
                        
                        c_star = optimizer(
                            obj_consumption_after_pay,
                            bc_min,
                            bc_max,
                            args=(par, sol_V, assets, s_lr, chi, t),
                            tol=par.opt_tol
                        )

                        sol_c[idx] = c_star
                        sol_a[idx] = assets + chi + s_lr - sol_c[idx]
                        sol_h[idx] = hours_place
                        sol_V[idx] = value_function_after_pay(par, sol_V, c_star, assets, s_lr, chi, t)

                    else:

                        if par.retirement_age <= t:
                            chi = retirement_payment(par, sol_V, assets, savings, s_lr, t)

                            bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, savings, human_capital_place, s_lr, chi, t)
                            
                            c_star = optimizer(
                                obj_consumption_under_pay,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, assets, savings, chi, t),
                                tol=par.opt_tol
                            )
                            s_retirement = savings / (1-(t-par.retirement_age)*(par.share_lr*(1/par.EL)+(1-par.share_lr)*(1/par.m)))
                            s_lr = par.share_lr * (s_retirement/par.EL)
                            s_rp = (1-par.share_lr) * (s_retirement/par.m)
                            sol_c[idx] = c_star 
                            sol_a[idx] = assets + chi + s_lr + s_rp - sol_c[idx]
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_function_under_pay(par, sol_V, c_star, assets, savings, chi, t)

                        else:

                            idx = (t, a_idx, s_idx, k_idx)
                            chi = 0.0 
                            
                            h_star = optimize_outer(
                                obj_hours,         # the hours objective
                                par.h_min,
                                par.h_max,
                                args=(par, sol_V, sol_EV, assets, savings, human_capital, s_lr, chi, t),
                                tol=par.opt_tol
                            )

                            bc_min, bc_max = budget_constraint(par, sol_V, assets, h_star, savings, human_capital, s_lr, chi, t)
                            c_star = optimizer(
                                obj_consumption,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                tol=par.opt_tol
                            )

                            sol_h[idx] = h_star
                            sol_c[idx] = c_star
                            sol_a[idx] = assets + (1-par.tau[t])*sol_h[idx]*wage(par, human_capital, t) - sol_c[idx]
                            sol_V[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)

    return sol_c, sol_a, sol_h, sol_V

