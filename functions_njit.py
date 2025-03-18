from numba import njit, prange
import numpy as np 

from consav.linear_interp import interp_1d, interp_2d, interp_3d
from optimizers import optimizer, optimize_outer, interp_3d_vec
from jit_module import jit_if_enabled


#######################################################################
# Structure 
#   1. Essentiel functions for such as utility, bequest, and wage
#   2. Value functions
#   3. Helper functions in solving and optimizing
#   4. Objective functions 
#   5. Solving the model
#######################################################################


# 1. Essentiel functions for such as utility, bequest, and wage 
@jit_if_enabled(fastmath=True)
def utility(par, c, h):
    return ((c+1)**(1-par.sigma))/(1-par.sigma) - par.work_cost*(h**(1+par.gamma))/(1+par.gamma)


@jit_if_enabled(fastmath=True)
def bequest(par, a):
    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)


@jit_if_enabled(fastmath=True)
def wage(par, k, t):
    return (1-par.upsilon)*par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)


@jit_if_enabled(fastmath=True)
def retirement_payment(par, a, s, t):
    base_payment = par.chi_base

    a_return = (par.r_a/(1+par.r_a)) * a

    s_lr, s_rp = calculate_retirement_payouts(par, s, t)

    if par.retirement_age +par.m< t:
        income = s_lr + a_return
    else: 
        income = s_lr + s_rp + a_return

    exceed = np.maximum(0, income - par.chi_max)
    extra_pension = np.maximum(0, par.chi_extra_start - exceed*par.rho)

    return (1-par.upsilon)*(base_payment + extra_pension)


# 2. Value functions
@jit_if_enabled(fastmath=True)
def value_function_after_pay(par, sol_V, c, a, s, t):

    hours = 0.0
    V_next = sol_V[t+1,:,0,0]

    s_lr, _ = calculate_retirement_payouts(par, s, t)
    chi = retirement_payment(par, a, s, t)

    a_next = (1+par.r_a)*(a + chi + s_lr - c)

    EV_next = interp_1d(par.a_grid, V_next, a_next)
    
    return utility(par, c, hours) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_function_under_pay(par, sol_V, c, a, s, t):

    hours = 0.0
    V_next = sol_V[t+1,:,:,0]
    
    s_lr, s_rp = calculate_retirement_payouts(par, s, t)
    chi = retirement_payment(par, a, s, t)

    a_next = (1+par.r_a)*(a + s_lr + s_rp + chi - c)
    s_next = s - s_lr - s_rp
    
    EV_next = interp_2d(par.a_grid,par.s_grid, V_next, a_next,s_next)

    return utility(par, c, hours) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_function_unemployed(par, sol_V, sol_EV, c, h, a, s, k, t):
    a_next = (1+par.r_a)*(a + par.benefit - c)
    s_next = (1+par.r_s)*s
    k_next = (1-par.delta)*k 

    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, t):

    a_next = (1+par.r_a)*(a + (1-par.tau[t])*h*wage(par, k, t) - c)
    s_next = (1+par.r_s)*(s + par.tau[t]*h*wage(par, k, t))
    k_next = ((1-par.delta)*k + h)

    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_next_period_after_reti(par, c, a, s, t):
    h = 0.0

    s_lr, _ = calculate_retirement_payouts(par, s, t)
    chi = retirement_payment(par, a, s, t)

    a_next = (1+par.r_a)*(a + chi + s_lr - c)

    return utility(par, c, h) + bequest(par, a_next)


# 3. Helper functions in solving and optimizing
@jit_if_enabled(fastmath=True)
def budget_constraint(par, h, a, s, k, ex, t):

    chi = retirement_payment(par, a, s, t)
    s_lr, s_rp = calculate_retirement_payouts(par, s, t)

    if t >= par.retirement_age + par.m:
        return par.c_min, max(par.c_min*2, a + chi + s_lr)
        
    elif par.retirement_age + par.m > t >= par.retirement_age:
        return par.c_min, max(par.c_min*2, a + s_lr + s_rp + chi)

    else:
        if ex==0:
            return par.c_min, max(par.c_min*2, a + par.benefit )
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
                    k_next = k_next*par.xi_v[idx]  
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
def obj_consumption_unemployed(c, par, sol_V, sol_EV, h, a, s, k, t):
    return -value_function_unemployed(par, sol_V, sol_EV, c, h, a, s, k, t)


@jit_if_enabled()
def obj_consumption_after_pay(c, par, sol_V, a, s, t):
    return -value_function_after_pay(par, sol_V, c, a, s, t)


@jit_if_enabled()
def obj_consumption_under_pay(c, par, sol_V, a, s, t):
    return -value_function_under_pay(par, sol_V, c, a, s, t)


@jit_if_enabled(fastmath=True)
def obj_hours(h, par, sol_V, sol_EV, a, s, k, ex, t, dist):

    bc_min, bc_max = budget_constraint(par, h, a, s, k, ex, t)
    
    c_star = optimizer(
        obj_consumption,     
        bc_min, 
        bc_max,
        args=(par, sol_V, sol_EV, h, a, s, k, t),
        tol=dist
    )
    
    val_at_c_star = -value_function(par, sol_V, sol_EV, c_star, h, a, s, k, t)
    
    return val_at_c_star


@jit_if_enabled(fastmath=True)
def calculate_retirement_payouts(par, savings, t):
    if t >= par.retirement_age + par.EL:
        s_retirement = -savings / (1 - ((t-par.retirement_age)/par.EL)*(par.share_lr)-(1-par.share_lr))
        s_lr = par.share_lr * (s_retirement/par.EL) 
        s_rp = (1-par.share_lr) * (s_retirement/par.m)
    
    elif t > par.retirement_age + par.m:
        s_retirement = savings / (1 - ((t-par.retirement_age)/par.EL)*(par.share_lr)-(1-par.share_lr))
        s_lr = par.share_lr * (s_retirement/par.EL) 
        s_rp = (1-par.share_lr) * (s_retirement/par.m)
    
    else:
        s_retirement = savings / (1 - (t-par.retirement_age)*(par.share_lr*(1/par.EL) + (1-par.share_lr)*(1/par.m)))
        s_lr = par.share_lr * (s_retirement/par.EL) 
        s_rp = (1-par.share_lr) * (s_retirement/par.m)
        
    return s_lr, s_rp


@jit_if_enabled(fastmath=True)
def calculate_last_period_consumption(par, assets, savings, t):
    chi = retirement_payment(par, assets, savings, t)
    s_lr, _ = calculate_retirement_payouts(par, savings, t)

    if par.mu != 0.0:
        # With bequest motive
        return max(((1/(1+(par.mu*(1+par.r_a))**(-1/par.sigma)*(1+par.r_a))) 
                    * (par.mu*(1+par.r_a))**(-1/par.sigma) 
                    * ((1+par.r_a)*(assets+chi+s_lr)+par.a_bar)), 0)
    
    else: 
        # No bequest motive
        return (assets + chi + s_lr)


# 5. Solving the model
@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol, do_print = False):

    human_capital_place, hours_place, ex_place = 0, 0, 0

    sol_a = sol.a
    sol_ex = sol.ex
    sol_c = sol.c
    sol_c_un = sol.c_un
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
                    
                    if t == par.T - 1: # Last period

                        sol_c[idx] = calculate_last_period_consumption(par, assets, savings, t)
                        sol_ex[idx] = ex_place
                        sol_h[idx] = hours_place
                        sol_V[idx] = value_next_period_after_reti(par, sol_c[idx], assets, savings, t)

                    elif t >= par.retirement_age + par.m: # After retirement age, with "livrente"

                        bc_min, bc_max = budget_constraint(par, hours_place, assets, savings, human_capital_place, ex_place, t)

                        c_star = optimizer(
                            obj_consumption_after_pay,
                            bc_min,
                            bc_max,
                            args=(par, sol_V, assets, savings, t),
                            tol=par.opt_tol
                        )

                        sol_c[idx] = c_star
                        sol_ex[idx] = ex_place
                        sol_h[idx] = hours_place
                        sol_V[idx] = value_function_after_pay(par, sol_V, c_star, assets, savings, t)

                    elif t >= par.retirement_age: # After retirement age, with "ratepension"

                        bc_min, bc_max = budget_constraint(par, hours_place, assets, savings, human_capital_place, ex_place, t)
                        
                        c_star = optimizer(
                            obj_consumption_under_pay,
                            bc_min,
                            bc_max,
                            args=(par, sol_V, assets, savings, t),
                            tol=par.opt_tol
                        )

                        sol_c[idx] = c_star 
                        sol_ex[idx] = ex_place
                        sol_h[idx] = hours_place
                        sol_V[idx] = value_function_under_pay(par, sol_V, c_star, assets, savings, t)

                    else: # Before retirement age
                        for ex in (0, 1):
                            if ex== 0: # Unemployed
                                h_unemployed = 0.0
                                bc_min, bc_max = budget_constraint(par, h_unemployed, assets, savings, human_capital, ex, t)
                                
                                c_star_u = optimizer(
                                    obj_consumption_unemployed,
                                    bc_min,
                                    bc_max,
                                    args=(par, sol_V, sol_EV, h_unemployed, assets, savings, human_capital, t),
                                    tol=par.opt_tol
                                )

                                V_unemployed = value_function_unemployed(par, sol_V, sol_EV, c_star_u, h_unemployed, assets, savings, human_capital, t)
                                sol_c_un[idx]  = c_star_u
                                
                            if ex== 1: # Employed
                                h_star = optimize_outer(
                                    obj_hours,       
                                    par.h_min,
                                    par.h_max,
                                    args=(par, sol_V, sol_EV, assets, savings, human_capital, ex, t),
                                    tol=par.opt_tol
                                )

                                bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, ex, t)
                                c_star = optimizer(
                                    obj_consumption,
                                    bc_min,
                                    bc_max,
                                    args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                    tol=par.opt_tol
                                )

                                V_employed = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)
                                sol_c[idx]  = c_star
                                sol_h[idx]  = h_star

                                if V_unemployed > V_employed:
                                    sol_V[idx]  = V_unemployed
                                    sol_ex[idx] = 0
                                else:
                                    sol_V[idx]  = V_employed
                                    sol_ex[idx] = 1

                    # Print if any sol_V is nan
                    # if np.isnan(sol_V[idx]):
                    #     print(f"NaN value at t={t}, a={assets}, s={savings}, k={human_capital}, ex={ex_place}")
                    #     assert False

    return sol_c, sol_c_un, sol_h, sol_ex, sol_V

# 6. simulation:
@jit_if_enabled(parallel=True)
def main_simulation_loop(par, sol, sim, do_print = False):

    sim_a = sim.a
    sim_s = sim.s
    sim_k = sim.k
    sim_c = sim.c
    sim_h = sim.h
    sim_w = sim.w
    sim_ex = sim.ex
    sim_a_init = sim.a_init
    sim_s_init = sim.s_init
    sim_k_init = sim.k_init
    sim_xi = sim.xi
    sim_s_lr_init = sim.s_lr_init
    sim_s_rp_init = sim.s_rp_init
    sim_chi_payment = sim.chi_payment
    
    sol_ex = sol.ex
    sol_c = sol.c
    sol_c_un = sol.c_un
    sol_h = sol.h

    # i. initialize states
    sim_a[:,0] = sim_a_init[:]
    sim_s[:,0] = sim_s_init[:]
    sim_k[:,0] = sim_k_init[:]

    for t in range(par.simT):

        # ii. interpolate optimal consumption and hours
        
        if t < par.retirement_age:
            interp_3d_vec(par.a_grid, par.s_grid, par.k_grid, sol_ex[t], sim_a[:,t], sim_s[:,t], sim_k[:,t], sim_ex[:,t])
            sim_ex[:,t] = np.maximum(0, np.round(sim_ex[:,t]))

            for i in prange(par.simN):
                if sim_ex[i,t] == 0: 
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c_un[t], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = 0.0
                    sim_w[i,t] = wage(par, sim_k[i,t], t)

                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] +par.benefit - sim_c[i,t])
                    sim_s[i,t+1] = (1+par.r_s)*sim_s[i,t]
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                else: 
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c[t], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_h[t], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + (1-par.tau[t])*sim_h[i,t]*sim_w[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = (1+par.r_s)*(sim_s[i,t] + par.tau[t]*sim_h[i,t]*sim_w[i,t])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t]


        # iii. store next-period states
        else:
            interp_3d_vec(par.a_grid, par.s_grid, par.k_grid, sol_c[t], sim_a[:,t], sim_s[:,t], sim_k[:,t], sim_c[:,t])
            interp_3d_vec(par.a_grid, par.s_grid, par.k_grid, sol_h[t], sim_a[:,t], sim_s[:,t], sim_k[:,t], sim_h[:,t])
            interp_3d_vec(par.a_grid, par.s_grid, par.k_grid, sol_ex[t], sim_a[:,t], sim_s[:,t], sim_k[:,t], sim_ex[:,t])
            sim_ex[:,t] = np.maximum(0, np.round(sim_ex[:,t]))


            if t == par.retirement_age:
                sim_s_lr_init[:] = (sim_s[:,t]/par.EL) * par.share_lr
                sim_s_rp_init[:] = (sim_s[:,t]/par.m) * (1-par.share_lr)

            if par.retirement_age <= t < par.retirement_age + par.m: 
                sim_chi_payment[:,t] = retirement_payment(par, sim_a[:,t], sim_s[:,t], t)
                sim_w[:,t] = wage(par, sim_k[:,t], t)
                sim_a[:,t+1] = (1+par.r_a)*(sim_a[:,t] + sim_s_lr_init[:] + sim_s_rp_init[:] + sim_chi_payment[:,t] - sim_c[:,t])
                sim_s[:,t+1] = np.maximum(0, sim_s[:,t] - (sim_s_lr_init[:] + sim_s_rp_init[:]))
                sim_k[:,t+1] = ((1-par.delta)*sim_k[:,t])*sim_xi[:,t]

            
            elif par.retirement_age + par.m <= t < par.T-1:
                sim_chi_payment[:,t] = retirement_payment(par, sim_a[:,t], sim_s[:,t], t)
                sim_w[:,t] = wage(par, sim_k[:,t], t)
                sim_a[:,t+1] = (1+par.r_a)*(sim_a[:,t] + sim_s_lr_init[:] + sim_chi_payment[:,t] - sim_c[:,t])
                sim_s[:,t+1] = np.maximum(0, sim_s[:,t] - sim_s_lr_init[:])
                sim_k[:,t+1] = ((1-par.delta)*sim_k[:,t])*sim_xi[:,t]
                
            else:
                sim_w[:,t] = wage(par, sim_k[:,t], t)

    return sim_a, sim_s, sim_k, sim_c, sim_h, sim_w, sim_ex, sim_chi_payment
