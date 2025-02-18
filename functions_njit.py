from numba import njit
import numpy as np 
from consav.linear_interp import interp_1d, interp_2d, interp_3d

USE_JIT = True  # Set to False to disable JIT for debugging


def jit_if_enabled(*args, **kwargs):
    """ Apply @njit only if USE_JIT is True """
    return njit(*args, **kwargs) if USE_JIT else (lambda f: f)


@jit_if_enabled()
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
