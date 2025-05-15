import numpy as np
from scipy.optimize import root_scalar
from functions_njit import *

# Welfare measurements for simulation 
def replacement_rate_fct(model):
    '''Can be used without policy changes'''
    par = model.par  
    sim = model.sim
    
    return np.mean(sim.income[:,par.start_before:par.end_before],axis=1) /sim.income[:,par.after_retirement]

def consumption_replacement_rate_fct(model):
    '''Can be used without policy changes'''
    par = model.par  
    sim = model.sim

    start_before = par.retirement_age-par.replacement_rate_bf_start
    end_before = par.retirement_age-par.replacement_rate_bf_end
    after_retirement = par.retirement_age +par.replacement_rate_af_start
    consumption_before =  sim.c[:,start_before:end_before].mean(axis=1)
    consumption_after = sim.c[:,after_retirement]

    return consumption_after/consumption_before

# Consumption equivalence 
def find_consumption_equivalence(original_model, new_model, do_print= False, the_method = 'brentq'):
    ''' Can be used to measure the impact of policy changes'''
    # c. Calculate welfare 
    EV_og = expected_lifetime_utility_distribution(original_model,  original_model.sim.c,   original_model.sim.h,   original_model.sim.a, original_model.sim.k)
    EV_new = expected_lifetime_utility_distribution(new_model,      new_model.sim.c,        new_model.sim.h,        new_model.sim.a, new_model.sim.k)   
    if do_print:
        print(f'Expected welfare  before parameter changes: {EV_og}')
        print(f'Expected welfare after parameter changes: {EV_new}')

    # d. Test bounds 
    # Set up bounds, and check if bounds result in target below and above 0, else update 
    phi_lower = -0.999
    phi_upper =  1.0  
    f_lower, f_upper = objective(phi_lower, EV_new, original_model), objective(phi_upper, EV_new, original_model)
    expansion_factor = 2.0
    while f_lower * f_upper > 0:  # No sign change → expand range
        # phi_lower *= expansion_factor
        phi_upper *= expansion_factor
        f_lower, f_upper = objective(phi_lower, EV_new, original_model), objective(phi_upper, EV_new, original_model)
        print(f"Expanding range: phi_lower={phi_lower}, phi_upper={phi_upper}")

        if abs(phi_lower) > 1e5 or abs(phi_upper) > 1e5:  # Prevent infinite expansion
            raise ValueError("Could not find a valid bracket for root-finding.")
    
    # e. Find root
    result = root_scalar(objective, bracket=[phi_lower, phi_upper], args=(EV_new , original_model), method=the_method)

    # f. analytical solution:
    phi_analytical = analytical_consumption_equivalence(original_model, EV_new)

    if result.converged:
        if do_print:
            print(f'Consumption at every age before the policy change must change with {round(result.root*100,1)} pct. to keep the same utility, and analytically: {round(phi_analytical*100,1)} pct.')
        return result.root
    else:
        raise ValueError("Root-finding for phi did not converge")
    
# Calculate elasticity of labor supply
def labor_elasticity(original_model, new_model):
    ''' Can be used to measure the impact of policy changes on labor supply'''
    # a. Original model
    par_og = original_model.par
    sim_og = original_model.sim

    # b. New model
    sim_new = new_model.sim
    
    # weights for the labor supply
    pi_cum = np.cumprod(par_og.pi)
    pi_weight = pi_cum/np.cumsum(pi_cum)
    
    # e. Calculate labor supply before and after
    # intensive margin
    sim_og_h_ex_1 = np.where(sim_og.ex == 1, sim_og.h, np.nan)
    sim_new_h_ex_1 = np.where(sim_new.ex == 1, sim_new.h, np.nan)
    sim_og_h = np.nanmean(sim_og_h_ex_1, axis=0)# age specific average 
    sim_new_h = np.nanmean(sim_new_h_ex_1, axis=0) # age specific average
    intensive_margin_age = (sim_new_h-sim_og_h)/sim_og_h 

    intensive_margin = np.nansum(pi_weight[:par_og.last_retirement] * intensive_margin_age[:par_og.last_retirement], axis=0)

    # extensive margin
    sim_og_ex = np.nansum(sim_og.ex, axis=0) # age specific average
    sim_new_ex = np.nansum(sim_new.ex, axis=0) # age specific average
    extensive_margin_age = (sim_new_ex-sim_og_ex)/par_og.simN
    print(sim_og_ex)
    print(sim_new_ex)
    
    extensive_margin = np.nansum(pi_weight[:par_og.last_retirement] * extensive_margin_age[:par_og.last_retirement], axis=0)

    # Total labor supply effect 
    total_margin_og = np.sum(sim_og.h, axis=0)
    total_margin_new = np.sum(sim_new.h, axis=0)
    total_margin_age = (total_margin_new-total_margin_og)/total_margin_og
    total_margin = np.nansum(pi_weight[:par_og.last_retirement] * total_margin_age[:par_og.last_retirement], axis=0)

    # total margin
    return intensive_margin, extensive_margin, total_margin, intensive_margin_age[:par_og.last_retirement], extensive_margin_age[:par_og.last_retirement], total_margin_age[:par_og.last_retirement]     

def kill_people(moment, model):
    import random

    killed_moment = moment.copy()

    for i in range(killed_moment.shape[0]):
        kill = 0
        for j in range(killed_moment.shape[1]):
            chance_of_death = random.random()
            if chance_of_death >= model.par.pi[j]:
                kill = 1

            if kill == 1:
                killed_moment[i, j] = np.nan
            else:
                pass

    return killed_moment


# Help functions
def bequest(model, a):
    '''Cannot be njited'''
    par = model.par
    if par.mu == 0.0:
        return np.zeros_like(a)
    else:
        return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

def utility_work(model, h, k):
    '''Cannot be njited'''
    par = model.par
    return -((par.zeta)/(1+k)) * (h**(1+par.gamma))/(1+par.gamma)

def utility_consumption(model, c):
    '''Cannot be njited'''
    par = model.par
    return (c**(1-par.sigma))/(1-par.sigma) 

# Expected welfare 
def expected_lifetime_utility_distribution(model, c, h, a, k):
    par = model.par
    N, T = c.shape

    beta_vector = par.beta**np.arange(T)
    beta_pi = beta_vector*par.pi
    beta_1_pi = beta_vector*(1-par.pi)

    uc = utility_consumption(model, c)
    uh = utility_work(model, h, k)
    bq = bequest(model, a)
    total_utility = 0.0
    for t in range(T):
        total_utility += beta_pi[t]* (np.nansum(uc[:,t])+ np.nansum(uh[:,t])) + beta_1_pi[t]  * np.nansum(bq[:,t])
    return total_utility/N

def expected_lifetime_utility_scaled(model, phi, c, h, a, k):
    return expected_lifetime_utility_distribution(model, (1.0 + phi)*c, h, a, k)

# analytical solution for consumption equivalence
def analytical_consumption_equivalence(original_model, EV_new):
    par_og = original_model.par
    sim_og = original_model.sim
    
    # Calculate individual utility
    uc = utility_consumption(original_model, sim_og.c) 
    uh = utility_work(original_model, sim_og.h, sim_og.k)
    bq = bequest(original_model, sim_og.a)

    # Calculate discount and probability of dying vector
    beta_vector = par_og.beta**np.arange(par_og.T)
    beta_pi = beta_vector*par_og.pi
    beta_1_pi = beta_vector*(1-par_og.pi)
    
    utility_work_bequest = 0.0
    utility_con  = 0.0
    for t in range(par_og.T):
        utility_work_bequest += beta_pi[t]* np.sum(uh[:,t]) + beta_1_pi[t]  * np.sum(bq[:,t])
        utility_con += beta_pi[t]*np.sum(uc[:,t])
    utility_work_bequest /= par_og.simN
    utility_con /= par_og.simN

    return ((EV_new-utility_work_bequest)/utility_con)**(1/(1-par_og.sigma))-1 


def make_new_model(model, theta, theta_names, do_print = False):
    ''' Can be used to measure the impact of policy changes'''
    # a. Original model
    original_model = model
    par_og = original_model.par

    # b. New model
    new_model = model.copy()

    # c. Overview of changes 
    if do_print:
        for idx, name in enumerate(theta_names):
            print(f'The original value of {name}: {par_og.__dict__[name]}, the new value will be: {theta[idx]}')

    # d. Update the new model with new parameters
    for i, name in enumerate(theta_names):
        setattr(new_model.par, name, theta[i])
    if do_print:
        print('Solving the new model')
    new_model.solve(do_print = do_print)
    if do_print:
        print('Simulating the new model')
    new_model.simulate()

    return original_model, new_model 

# f. Create objective function
def objective(phi, new_EV, original_model):
    c = original_model.sim.c
    h = original_model.sim.h
    a = original_model.sim.a
    k = original_model.sim.k

    compensate_EV = expected_lifetime_utility_scaled(original_model, phi, c, h, a, k)
    
    return new_EV - compensate_EV

