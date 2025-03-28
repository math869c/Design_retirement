import numpy as np
from scipy.optimize import root_scalar

# Welfare measurements for simulation 
def replacement_rate_fct(model):
    '''Can be used without policy changes'''
    par = model.par  
    sim = model.sim

    start_before = par.retirement_age-par.replacement_rate_bf_start
    end_before = par.retirement_age-par.replacement_rate_bf_end
    after_retirement = par.retirement_age +par.replacement_rate_af_start
    income_before =  ((1-par.tau[start_before:end_before])*sim.h[:,start_before:end_before]*sim.w[:, start_before:end_before] +
        (par.r_a/(1+par.r_a))* sim.a[:,start_before:end_before]).mean(axis=1) 
    income_after = sim.s_lr_init[:] + sim.s_rp_init[:] + sim.chi_payment[:,after_retirement] +\
        (par.r_a/(1+par.r_a))* sim.a[:,after_retirement]

    return income_after/income_before

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

# Help functions
def bequest(model, a):
    '''Cannot be njited'''
    par = model.par
    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

def utility_work(model, h):
    '''Cannot be njited'''
    par = model.par
    return - par.work_cost*(h**(1+par.gamma))/(1+par.gamma)

def utility_consumption(model, c):
    '''Cannot be njited'''
    par = model.par
    return (c**(1-par.sigma))/(1-par.sigma) 

# Expected welfare 
def expected_lifetime_utility_distribution(model, c, h, a):
    par = model.par
    N, T = c.shape

    beta_vector = par.beta**np.arange(T)
    beta_pi = beta_vector*par.pi
    beta_1_pi = beta_vector*(1-par.pi)

    uc = utility_consumption(model, c)
    uh = utility_work(model, h)
    bq = bequest(model, a)
    total_utility = 0.0
    for t in range(T):
        total_utility += beta_pi[t]* (np.sum(uc[:,t])+ np.sum(uh[:,t])) + beta_1_pi[t]  * np.sum(bq[:,t])
    return total_utility/N

def expected_lifetime_utility_scaled(model, phi, c, h, a):
    return expected_lifetime_utility_distribution(model, (1.0 + phi)*c, h, a)

# analytical solution
def analytical_consumption_equivalence(original_model, EV_new):
    par_og = original_model.par
    sim_og = original_model.sim
    
    # Calculate individual utility
    uc = utility_consumption(original_model, sim_og.c) 
    uh = utility_work(original_model, sim_og.h)
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


# Consumption equivalence 
def find_consumption_equivalence(model, theta, theta_names, do_print= False, the_method = 'brentq'):
    ''' Can be used to measure the impact of policy changes'''
    # a. Original model
    original_model = model
    par_og = original_model.par
    sim_og = original_model.sim

    # b. New model
    new_model = model.copy()
    par_new = new_model.par
    sim_new = new_model.sim

    # c. Overview of changes 
    if do_print:
        for idx, name in enumerate(theta_names):
            print(f'The original value of {name}: {par_og.__dict__[name]}, the new value will be: {theta[idx]}')

    # d. Update the new model with new parameters
    for i, name in enumerate(theta_names):
        setattr(new_model.par, name, theta[i])
    print('Solving the model')
    new_model.solve()
    print('Simulating the model')
    new_model.simulate()

    # e. Calculate welfare 
    EV_og = expected_lifetime_utility_distribution(original_model, sim_og.c,   sim_og.h,   sim_og.a)
    EV_new = expected_lifetime_utility_distribution(new_model,     sim_new.c,  sim_new.h,  sim_new.a)
    if do_print:
        print(f'Consumption utility before parameter changes: {EV_og}')
        print(f'Consumption utility after parameter changes: {EV_new}')

    # f. Create objective function
    def objective(phi, new_EV, c, h, a):
        compensate_EV = expected_lifetime_utility_scaled(original_model, phi, c, h, a)
        return new_EV - compensate_EV
    
    # g. Test bounds 
    # Set up bounds, and check if bounds result in target below and above 0, else update 
    phi_lower = -0.5
    phi_upper =  1.0  
    f_lower, f_upper = objective(phi_lower, EV_new, sim_og.c,   sim_og.h,   sim_og.a), objective(phi_upper, EV_new, sim_og.c,   sim_og.h,   sim_og.a)
    expansion_factor = 2.0
    while f_lower * f_upper > 0:  # No sign change → expand range
        phi_lower /= expansion_factor
        phi_upper *= expansion_factor
        f_lower, f_upper = objective(phi_lower, EV_new, sim_og.c,   sim_og.h,   sim_og.a), objective(phi_upper, EV_new, sim_og.c,   sim_og.h,   sim_og.a)
        print(f"Expanding range: phi_lower={phi_lower}, phi_upper={phi_upper}")

        if abs(phi_lower) > 1e5 or abs(phi_upper) > 1e5:  # Prevent infinite expansion
            raise ValueError("Could not find a valid bracket for root-finding.")
    
    # h. Find root
    result = root_scalar(objective, bracket=[phi_lower, phi_upper], args=(EV_new , sim_og.c,   sim_og.h,   sim_og.a), method=the_method)

    # i. analytical solution:
    phi_analytical = analytical_consumption_equivalence(original_model, EV_new)

    if result.converged:
        if do_print:
            print(f'Consumption at every age before the policy change must change with {round(result.root*100,1)} pct. to keep the same utility, and analytically: {round(phi_analytical*100,1)} pct.')
        return result.root
    else:
        raise ValueError("Root-finding for phi did not converge")