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
    return ((c+1)**(1-par.sigma))/(1-par.sigma) - (h**(1+par.gamma))/(1+par.gamma)


@jit_if_enabled(fastmath=True)
def bequest(par, a):
    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)


@jit_if_enabled(fastmath=True)
def wage(par, k, t):
    return (1-par.upsilon)*par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)

@jit_if_enabled(fastmath=True)
def income_fct(par, a, s, k, h, retirement_age, t):
    a_return = (par.r_a/(1+par.r_a)) * a

    s_lr, s_rp = calculate_retirement_payouts(par, s, retirement_age, t)

    if t >= retirement_age + par.m:
        return s_lr + a_return
    elif retirement_age <= t<retirement_age + par.m: 
        return s_lr + s_rp + a_return
    else:
        return (1-par.tau[t])*h*wage(par, k, t)  + a_return



@jit_if_enabled(fastmath=True)
def retirement_payment(par, income, t):
    base_payment = par.chi_base

    exceed = np.maximum(0, income - par.chi_max)
    extra_pension = np.maximum(0, par.chi_extra_start - exceed*par.rho)

    if t >= par.retirement_age:
        return (1-par.upsilon)*(base_payment + extra_pension)
    else:
        return 0.0


# 2. Value functions
@jit_if_enabled(fastmath=True)
def value_function_after_retirement(par, sol_V, c, a, s, retirement_age, t):

    retirement_age_idx = retirement_age - 30
    hours = 0.0
    V_next = sol_V[t+1, :, :, 0, retirement_age_idx]

    income = income_fct(par, a, s, 0.0, 0.0, retirement_age, t)

    s_lr, s_rp = calculate_retirement_payouts(par, s, retirement_age, t)
    chi = retirement_payment(par, income, t)

    if t >= retirement_age + par.m:
        a_next = (1+par.r_a)*(a + s_lr + chi - c)
    else:
        a_next = (1+par.r_a)*(a + s_lr + s_rp + chi - c)

    s_next = s
    
    EV_next = interp_2d(par.a_grid, par.s_grid, V_next, a_next, s_next)

    return utility(par, c, hours) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, t):

    a_next = (1+par.r_a)*(a + (1-par.tau[t])*h*wage(par, k, t) - c)
    s_next = (1+par.r_s)*(s + par.tau[t]*h*wage(par, k, t))
    k_next = ((1-par.delta)*k + h)

    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)



@jit_if_enabled(fastmath=True)
def value_last_period(par, c, a, s, retirement_age, t):
    h = 0.0

    s_lr, _ = calculate_retirement_payouts(par, s, retirement_age, t)
    
    income = income_fct(par, a, s, 0.0, 0.0, retirement_age, t)
    chi = retirement_payment(par, income, t)

    a_next = (1+par.r_a)*(a + chi + s_lr - c)

    return utility(par, c, h) + bequest(par, a_next)


# 3. Helper functions in solving and optimizing
@jit_if_enabled(fastmath=True)
def budget_constraint(par, h, a, s, k, retirement_age, ex, t):

    income = income_fct(par, a, s, k, h, retirement_age, t)
    chi = retirement_payment(par, income, t)
    s_lr, s_rp = calculate_retirement_payouts(par, s, retirement_age, t)

    if t >= retirement_age + par.m:
        return par.c_min, max(par.c_min*2, a + chi + s_lr)
        
    elif t > retirement_age:
        return par.c_min, max(par.c_min*2, a + chi + s_lr + s_rp)
    
    elif t == retirement_age:
        if ex==0:
            return par.c_min, max(par.c_min*2, a + chi + s_lr + s_rp)
        else:
            return par.c_min, max(par.c_min*2, a + (1-par.tau[t])*h*wage(par, k, t) + chi)

    else:
        return par.c_min, max(par.c_min*2, a + (1-par.tau[t])*h*wage(par, k, t) + chi)


@jit_if_enabled(fastmath=True)
def precompute_EV_next(par, sol_V, retirement_idx, t):

    V_next = sol_V[t+1, :, :, :, retirement_idx]

    EV = np.zeros((len(par.a_grid), len(par.s_grid), len(par.k_grid)))

    for i_a, a_next in enumerate(par.a_grid):
        for i_s, s_next in enumerate(par.s_grid):
            for i_k, k_next in enumerate(par.k_grid):

                EV_val = 0.0
                for idx in range(par.N_xi):
                    k_temp_ = k_next*par.xi_v[idx] 
                    V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid, V_next, a_next, s_next, k_temp_)
                    EV_val += V_next_interp * par.xi_p[idx]

                EV[i_a, i_s, i_k] = EV_val

    return EV


@jit_if_enabled(fastmath=True)
def calculate_retirement_payouts(par, savings, retirement_age, t):
    if t >= retirement_age + par.m:
        s_retirement = savings
        s_lr = par.share_lr * (s_retirement/par.EL) 

        return s_lr, np.zeros_like(s_lr) +np.nan
    
    else:
        s_retirement = savings
        s_lr = par.share_lr * (s_retirement/par.EL) 
        s_rp = (1-par.share_lr) * (s_retirement/par.m) 
    
        return s_lr, s_rp

@jit_if_enabled(fastmath=True)
def calculate_last_period_consumption(par, a, s, retirement_age, t):
    income = income_fct(par, a, s, 0.0, 0.0, retirement_age, t)
    chi = retirement_payment(par, income, t)
    s_lr, _ = calculate_retirement_payouts(par, s, retirement_age, t)

    if par.mu != 0.0:
        # With bequest motive
        return max(((1/(1+(par.mu*(1+par.r_a))**(-1/par.sigma)*(1+par.r_a))) 
                    * (par.mu*(1+par.r_a))**(-1/par.sigma) 
                    * ((1+par.r_a)*(a+chi+s_lr)+par.a_bar)), 0)
    
    else: 
        # No bequest motive
        return (a + chi + s_lr)


# 4. Objective functions 
@jit_if_enabled(fastmath=True)
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, t)


@jit_if_enabled()
def obj_consumption_after_retirement(c, par, sol_V, a, s, retirement_age, t):
    return -value_function_after_retirement(par, sol_V, c, a, s, retirement_age, t)



@jit_if_enabled(fastmath=True)
def obj_hours(h, par, sol_V, sol_EV, a, s, k, retirement_age, ex, t, dist):

    bc_min, bc_max = budget_constraint(par, h, a, s, k, retirement_age, ex, t)
    
    c_star = optimizer(
        obj_consumption,     
        bc_min, 
        bc_max,
        args=(par, sol_V, sol_EV, h, a, s, k, t),
        tol=dist
    )
    
    val_at_c_star = -value_function(par, sol_V, sol_EV, c_star, h, a, s, k, t)
    
    return val_at_c_star



# 5. Solving the model
@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol, do_print = False):

    human_capital_place, hours_place, ex_place = 0.0, 0.0, 0.0

    sol_a = sol.a
    sol_ex = sol.ex
    sol_c = sol.c
    sol_c_un = sol.c_un
    sol_h = sol.h
    sol_V = sol.V
    V_employed = sol.V_employed
    V_unemployed = sol.V_unemployed
    
    for t in range(par.T - 1, -1, -1):
        if do_print:
            print(f"We are in t = {t}")


        retirement_ages = np.arange(par.first_retirement, min(par.last_retirement + 1, t + 1)) \
                        if t >= par.first_retirement + 1 else np.arange(par.first_retirement, par.first_retirement + 1)

        for retirement_age_idx, retirement_age in enumerate(retirement_ages):

            if t <= retirement_age:
                sol_EV = precompute_EV_next(par, sol_V, retirement_age_idx, t)

            for a_idx in prange(len(par.a_grid)):
                assets = par.a_grid[a_idx]

                for s_idx in range(len(par.s_grid)):
                    savings = par.s_grid[s_idx]

                    for k_idx in range(len(par.k_grid)):
                        human_capital = par.k_grid[k_idx]

                        idx = (t, a_idx, s_idx, k_idx, retirement_age_idx)
                        idx_next = (t+1, a_idx, s_idx, k_idx, retirement_age_idx)
                        
                        if t == par.T - 1: # Last period

                            sol_c[idx] = calculate_last_period_consumption(par, assets, savings, retirement_age, t)
                            sol_ex[idx] = ex_place
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_last_period(par, sol_c[idx], assets, savings, retirement_age, t)

                        elif t > retirement_age: # After retirement age, with "ratepension"

                            bc_min, bc_max = budget_constraint(par, hours_place, assets, savings, human_capital_place, retirement_age, ex_place, t)

                            c_star = optimizer(
                                obj_consumption_after_retirement,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, assets, savings, retirement_age, t),
                                tol=par.opt_tol
                            )

                            sol_c[idx] = c_star
                            sol_ex[idx] = ex_place
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_function_after_retirement(par, sol_V, c_star, assets, savings, retirement_age, t)
                   

                        elif t == retirement_age and sol_ex[idx_next] == 0.0:  
                            for ex in (0, 1):
                                if ex == 0.0: # Unemployed
                                    bc_min, bc_max = budget_constraint(par, hours_place, assets, savings, human_capital, retirement_age, ex, t)

                                    c_star_u = optimizer(
                                        obj_consumption_after_retirement,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, assets, savings, retirement_age, t),
                                        tol=par.opt_tol
                                    )

                                    V_unemployed[idx] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, retirement_age, t)
                                    sol_c_un[idx]  = c_star_u

                                if ex== 1: # Employed
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, retirement_age, ex, t),
                                        tol=par.opt_tol
                                    )

                                    bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, retirement_age, ex, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                        tol=par.opt_tol
                                    )

                                    V_employed[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)

                                    sol_c[idx]  = c_star
                                    sol_h[idx]  = h_star

                                    if V_unemployed[idx] > V_employed[idx]:
                                        sol_V[idx]  = V_unemployed[idx]
                                        sol_ex[idx] = 0.0
                                    else:
                                        sol_V[idx]  = V_employed[idx]
                                        sol_ex[idx] = 1.0

                        else:
                            h_star = optimize_outer(
                                obj_hours,       
                                par.h_min,
                                par.h_max,
                                args=(par, sol_V, sol_EV, assets, savings, human_capital, retirement_age, ex, t),
                                tol=par.opt_tol
                            )

                            bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, retirement_age, ex, t)
                            c_star = optimizer(
                                obj_consumption,
                                bc_min,
                                bc_max,
                                args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                tol=par.opt_tol
                            )

                            sol_ex[idx] = 1.0
                            sol_V[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)
                            sol_c[idx]  = c_star
                            sol_h[idx]  = h_star

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
    sim_income = sim.income
    sim_xi = sim.xi
    s_retirement = sim.s_retirement
    retirement_age = sim.retirement_age
    retirement_age_idx = sim.retirement_age_idx
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
        if t < 30:
            for i in prange(par.simN):
                sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c[t,:,:,:,0], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_h[t,:,:,:,0], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                sim_w[i,t] = wage(par, sim_k[i,t], t)
                sim_ex[i,t] = 1.0
            
                sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + (1-par.tau[t])*sim_h[i,t]*sim_w[i,t] - sim_c[i,t])
                sim_s[i,t+1] = (1+par.r_s)*(sim_s[i,t] + par.tau[t]*sim_h[i,t]*sim_w[i,t])
                sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t]

        elif t <= par.last_retirement:
            for i in prange(par.simN):
                if sim_ex[i,t-1] == 1.0:
                    sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_ex[t,:,:,:,t-par.first_retirement], sim_a[i,t], sim_s[i,t], sim_k[i,t]) 
                    sim_ex[i,t] = np.maximum(0, np.round(sim_ex[i,t]))
                else:
                    sim_ex[i,t] = 0.0

                if (sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 1.0) or (sim_ex[i,t-1] == 1.0 and t == par.last_retirement): 
                    retirement_age[i] = t
                    retirement_age_idx[i] = t - par.first_retirement
                    s_retirement[i] = sim_s[i,t]
                    sim_s_lr_init[i] = (s_retirement[i]/par.EL) * par.share_lr
                    sim_s_rp_init[i] = (s_retirement[i]/par.m) * (1-par.share_lr)

                    sim_h[i,t] = 0.0
                    sim_income[i] = income_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    sim_chi_payment[i,t] = retirement_payment(par, sim_income[i], t)

                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c_un[t,:,:,:,int(retirement_age_idx[i])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_w[i,t] = wage(par, sim_k[i,t], t)

                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_s_lr_init[i] + sim_s_rp_init[i] + sim_chi_payment[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                elif sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 0.0: 
                    sim_h[i,t] = 0.0
                    sim_income[i] = income_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    sim_chi_payment[i,t] = retirement_payment(par, sim_income[i], t)

                    sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i])], sim_a[i,t], s_retirement[i])
                    sim_w[i,t] = wage(par, sim_k[i,t], t)

                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_s_lr_init[i] + sim_s_rp_init[i] + sim_chi_payment[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                else: 
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_h[t,:,:,:,t-par.first_retirement], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_income[i] = income_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    sim_chi_payment[i,t] = retirement_payment(par, sim_income[i], t)

                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c[t,:,:,:,t-par.first_retirement], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_w[i,t] = wage(par, sim_k[i,t], t)

                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + (1-par.tau[t])*sim_h[i,t]*sim_w[i,t] + sim_chi_payment[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = (1+par.r_s)*(sim_s[i,t] + par.tau[t]*sim_h[i,t]*sim_w[i,t])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t]          

        elif t > par.last_retirement:
            sim_ex[:,t] = 0.0

            for i in prange(par.simN):
                sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i])], sim_a[i,t], s_retirement[i])
                sim_h[i,t] = 0.0

                if t < retirement_age[i] + par.m: 
                    sim_income[i] = income_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    sim_chi_payment[i,t] = retirement_payment(par, sim_income[i], t)

                    sim_w[i,t] = wage(par, sim_k[i,t], t)

                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_s_lr_init[i] + sim_s_rp_init[i] + sim_chi_payment[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                elif par.T - 1 > t >= retirement_age[i] + par.m:
                    sim_income[i] = income_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    sim_chi_payment[i,t] = retirement_payment(par, sim_income[i], t)

                    sim_w[i,t] = wage(par, sim_k[i,t], t)

                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_s_lr_init[i] + sim_chi_payment[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - sim_s_lr_init[i]
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]
                    
                else:
                    sim_w[i,t] = wage(par, sim_k[i,t], t)


    return sim_a, sim_s, sim_k, sim_c, sim_h, sim_w, sim_ex, sim_chi_payment
