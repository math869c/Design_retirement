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
def utility(par, sol_V,  c, h):

    return (c)**(1-par.sigma)/(1-par.sigma) - par.work_cost*(h)**(1+par.gamma)/(1+par.gamma)

@jit_if_enabled(fastmath=True)
def bequest(par, sol_V,  a):

    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled(fastmath=True)
def wage(par, sol_V,  k, t):

    return (1-par.upsilon)*par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)

# 2. Value functions
@jit_if_enabled(fastmath=True)
def value_function_after_pay(par, sol_V,  c, a, s_lr, t):

    hours = 0.0
    V_next = sol_V[t+1,:,0,0]
    a_next = (1+par.r_a)*(a + par.chi[t] + s_lr - c)
    EV_next = interp_1d(par.a_grid, V_next, a_next)
    
    
    return utility(par, sol_V, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled(fastmath=True)
def value_function_under_pay(par, sol_V,  c, a, s, t):

    hours = 0.0
    V_next = sol_V[t+1,:,:,0]
    s_retirement = s / (1-(t-par.retirement_age)*(par.share_lr*(1/par.EL)+(1-par.share_lr)*(1/par.m)))
    s_lr = par.share_lr * (s_retirement/par.EL)
    s_rp = (1-par.share_lr) * (s_retirement/par.m)
    # s_retirement = (par.m/(par.m-(t-par.retirement_age))) * s # skaleres op for den oprindelige s, naar man gaar på pension.
    a_next = (1+par.r_a)*(a + s_lr + s_rp + par.chi[t] - c)
    s_next = s - s_lr - s_rp
    
    EV_next = interp_2d(par.a_grid,par.s_grid, V_next, a_next,s_next)


    return utility(par, sol_V, c, hours) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled(fastmath=True)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, t):

    a_next = (1+par.r_a)*(a + (1-par.tau[t])*h*wage(par, sol_V, k, t) - c)
    s_next = (1+par.r_s)*(s + par.tau[t]*h*wage(par, sol_V, k, t))
    k_next = ((1-par.delta)*k + h)

    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, sol_V, c, h) + (1-par.pi[t+1])*par.beta*EV_next + par.pi[t+1]*bequest(par, sol_V, a_next)

@jit_if_enabled(fastmath=True)
def value_next_period_after_reti(par, sol_V, c, a, t):
    h = 0.0

    a_next = (1+par.r_a)*(a + par.chi[t] - c)
    

    return utility(par, sol_V, c, h) + bequest(par, sol_V, a_next)

# 3. Helping functions in solving and optimizing
@jit_if_enabled(fastmath=True)
def budget_constraint(par, sol_V, a, h, s, k, t):

    if par.retirement_age + par.m <= t:
        s_lr = s
        return par.c_min, max(par.c_min*2, a + par.chi[t] + s_lr)
    
    elif par.retirement_age <= t < par.retirement_age + par.m:
        s_retirement = s / (1-(t-par.retirement_age)*((2/3)*(1/par.EL)+(1/3)*(1/par.m)))
        s_lr = par.share_lr * (s_retirement/par.EL)
        s_rp = (1-par.share_lr) * (s_retirement/par.m)
        return par.c_min, max(par.c_min*2, a + s_lr + s_rp  + par.chi[t])

    else:
        return par.c_min, max(par.c_min*2, a + (1-par.tau[t])*h*wage(par, sol_V, k, t))

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
def obj_consumption_last_period(c, par, sol_V, a, t):
    """ negative of value_function_after_pay(par,sol_V,c,a,t) """
    return -value_next_period_after_reti(par, sol_V, c, a, t)

@jit_if_enabled()
def obj_consumption_after_pay(c, par, sol_V, a, s_lr, t):
    """ negative of value_function_after_pay(par,sol_V,c,a,t) """
    return -value_function_after_pay(par, sol_V, c, a, s_lr, t)

@jit_if_enabled()
def obj_consumption_under_pay(c, par, sol_V, a, s, t):
    """ negative of value_function_under_pay(par,sol_V,c,a,s,t) """
    return -value_function_under_pay(par, sol_V, c, a, s, t)



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
                    idx_next = (t+1, a_idx, s_idx, k_idx)

                    if t == par.T - 1:
                        # Analytical solution in the last period
                        if par.mu != 0.0:
                            # With bequest motive
                            s_retirement = savings / (1 - ((t-par.retirement_age)/par.EL)*(par.share_lr)-(1-par.share_lr))
                            s_lr = par.share_lr * (s_retirement/par.EL) 

                            sol_c[idx] = (1/(1+(par.mu*(1+par.r_a))**(-1/par.sigma)*(1+par.r_a))) * (par.mu*(1+par.r_a))**(-1/par.sigma) * ((1+par.r_a)*(assets+par.chi[t]+s_lr)+par.a_bar)
                            sol_a[idx] = assets + par.a_bar + s_lr + par.chi[t] - sol_c[idx]
                            # if assets +par.chi[t] - sol_c[idx] < 0:

                            # bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, savings, human_capital_place, t)

                            # c_star = optimizer(
                            #     obj_consumption_last_period,
                            #     bc_min,
                            #     bc_max,
                            #     args=(par, sol_V, assets, t),
                            #     tol=par.opt_tol
                            # )

                            # sol_c[idx] = c_star
                            # sol_a[idx] = assets + par.chi[t] + par.a_bar - sol_c[idx]
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx], assets, t)

                        else: 
                            # No bequest motive                          
                            s_retirement = savings / (1 - ((t-par.retirement_age)/par.EL)*par.share_lr-(1-par.share_lr))
                            s_lr = par.share_lr * (s_retirement/par.EL)
                            sol_c[idx] = (1+par.r_a)*(assets+par.chi[t] + s_lr)
                            sol_a[idx] = assets + par.chi[t] + par.a_bar - sol_c[idx]
                            sol_h[idx] = hours_place
                            sol_V[idx] = value_next_period_after_reti(par, sol_V, sol_c[idx], assets, t)

                    elif par.retirement_age +par.m <= t:

                        bc_min, bc_max = budget_constraint(par, sol_V, assets, hours_place, s_lr, human_capital_place, t) #er s_lr ok her i stedet for svaings? Det virker, bare lidt grimt
                        
                        c_star = optimizer(
                            obj_consumption_after_pay,
                            bc_min,
                            bc_max,
                            args=(par, sol_V, assets, s_lr, t),
                            tol=par.opt_tol
                        )

                        sol_c[idx] = c_star
                        sol_a[idx] = assets + par.chi[t] + s_lr - sol_c[idx]
                        sol_h[idx] = hours_place
                        sol_V[idx] = value_function_after_pay(par, sol_V, c_star, assets, s_lr, t)

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
                            s_retirement = savings / (1-(t-par.retirement_age)*(par.share_lr*(1/par.EL)+(1-par.share_lr)*(1/par.m)))
                            s_lr = par.share_lr * (s_retirement/par.EL)
                            s_rp = (1-par.share_lr) * (s_retirement/par.m)
                            sol_c[idx] = c_star 
                            sol_a[idx] = assets + par.chi[t] + s_lr + s_rp - sol_c[idx]
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
                            sol_a[idx] = assets + (1-par.tau[t])*sol_h[idx]*wage(par, sol_V, human_capital, t) - sol_c[idx]
                            sol_V[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)

    return sol_c, sol_a, sol_h, sol_V


# 6. simulate the model
@jit_if_enabled(fastmath=True)
def main_simulate_loop(par, sol, sim, do_print = False):
    # Init simulations
    sim.a[:,0] = sim.a_init[:]
    sim.s[:,0] = sim.s_init[:]
    sim.k[:,0] = sim.k_init[:]


    for t in range(par.simT):

        # ii. interpolate optimal consumption and hours
        interp_3d_vec(par.a_grid, par.s_grid, par.k_grid, sol.c[t], sim.a[:,t], sim.s[:,t], sim.k[:,t], sim.c[:,t])
        interp_3d_vec(par.a_grid, par.s_grid, par.k_grid, sol.h[t], sim.a[:,t], sim.s[:,t], sim.k[:,t], sim.h[:,t])
        if t == par.retirement_age:
            sim.s_lr_init[:] = (sim.s[:,t]/par.EL) * par.share_lr
            sim.s_rp_init[:] = (sim.s[:,t]/par.m) * (1-par.share_lr)

        # iii. store next-period states
        if t < par.retirement_age:
            # if t == 0:
            #     sim.w[:,t] = sim.w_init[:]*par.full_time_hours*sim.h[:,t]
            # else:
            sim.w[:,t] = wage(par, sol, sim.k[:,t], t)
            sim.a[:,t+1] = (1+par.r_a)*(sim.a[:,t] + (1-par.tau[t])*sim.h[:,t]*sim.w[:,t] - sim.c[:,t])
            sim.s[:,t+1] = (1+par.r_s)*(sim.s[:,t] + par.tau[t]*sim.h[:,t]*sim.w[:,t])
            sim.k[:,t+1] = ((1-par.delta)*sim.k[:,t] + sim.h[:,t])*sim.xi[:,t]

        elif par.retirement_age <= t < par.retirement_age + par.m: 
            sim.w[:,t] = wage(par, sol, sim.k[:,t], t)
            sim.a[:,t+1] = (1+par.r_a)*(sim.a[:,t] + sim.s_lr_init[:] + sim.s_rp_init[:] + par.chi[t] - sim.c[:,t])
            sim.s[:,t+1] = sim.s[:,t] - (sim.s_lr_init[:] + sim.s_rp_init[:])
            sim.k[:,t+1] = ((1-par.delta)*sim.k[:,t])*sim.xi[:,t]
        
        elif par.retirement_age + par.m <= t < par.T-1:
            sim.w[:,t] = wage(par, sol, sim.k[:,t], t)
            sim.a[:,t+1] = (1+par.r_a)*(sim.a[:,t] + sim.s_lr_init[:] + par.chi[t] - sim.c[:,t])
            sim.s[:,t+1] = sim.s[:,t] - sim.s_lr_init[:]
            sim.k[:,t+1] = ((1-par.delta)*sim.k[:,t])*sim.xi[:,t]
        
        else:
            sim.w[:,t] = wage(par, sol, sim.k[:,t], t)

    return sim.a, sim.c, sim.h, sim.s, sim.k, sim.w
