import numpy as np
from scipy.optimize import root_scalar

# Welfare measurements for simulation 
def replacement_rate_fct(self):
    '''Can be used without policy changes'''
    par = self.par  
    sim = self.sim

    start_before = par.retirement_age-par.replacement_rate_bf_start
    end_before = par.retirement_age-par.replacement_rate_bf_end
    after_retirement = par.retirement_age +par.replacement_rate_af_start
    income_before =  ((1-par.tau[start_before:end_before])*sim.h[:,start_before:end_before]*sim.w[:, start_before:end_before] +\
        (par.r_a/(1+par.r_a))* sim.a[:,start_before:end_before]).mean(axis=1) 
    income_after = sim.s_lr_init[:] + sim.s_rp_init[:] + sim.chi_payment[:] +\
        (par.r_a/(1+par.r_a))* sim.a[:,after_retirement]

    return income_after/income_before

def consumption_replacement_rate_fct(self):
    '''Can be used without policy changes'''
    par = self.par  
    sim = self.sim

    start_before = par.retirement_age-par.replacement_rate_bf_start
    end_before = par.retirement_age-par.replacement_rate_bf_end
    after_retirement = par.retirement_age +par.replacement_rate_af_start
    consumption_before =  sim.c[:,start_before:end_before].mean(axis=1)
    consumption_after = sim.c[:,after_retirement]

    return consumption_after/consumption_before

def expected_lifetime_utility(self, utility_matrix):
    '''Can be used without policy changes'''
    par = self.par  
    sim = self.sim
    
    beta_vector = par.beta**np.arange(par.T)
    beta_pi = beta_vector*par.pi
    return (utility_matrix@beta_pi).mean()

def utility_consumption(self, par, c):
    return (c**(1-par.sigma))/(1-par.sigma)

def find_consumption_equivalence(self, theta, theta_names, do_print= False, the_method = 'brentq'):
    ''' Can be used to measure the impact of policy changes'''
    par = self.par
    sim = self.sim
    
    # overview of changes 
    if do_print:
        for idx, name in enumerate(theta_names):
            print(f'The original value of {name}: {par.__dict__[name]}, the new value will be: {theta[idx]}')
        print(f'Consumption utility before parameter changes: {sim.EV[0]}')

    # Set previous consumption, so it does not update with new values
    avg_c_bf = sim.c[:,:].mean(axis=0).copy() 
    phi_lower = -0.5
    phi_upper =  1.0  

    # Utility after policy change
    for i, name in enumerate(theta_names):
        setattr(self.par, name, theta[i])
    self.solve()
    self.simulate()
    new_EV = sim.EV.copy()
    if do_print:
        print(f'Consumption utility after parameter changes: {sim.EV[0]}')

    def objective(phi, avg_c_bf, new_EV):
        utility_matrix_compensate = self.utility_consumption(par, (1 + phi) * avg_c_bf)
        utility_compensate = self.expected_lifetime_utility(utility_matrix_compensate)

        return new_EV - utility_compensate
    
    # Check if bounds are below and above 0, else update 
    f_lower, f_upper = objective(phi_lower, avg_c_bf, new_EV), objective(phi_upper, avg_c_bf, new_EV)
    expansion_factor = 2.0
    while f_lower * f_upper > 0:  # No sign change → expand range
        phi_lower /= expansion_factor
        phi_upper *= expansion_factor
        f_lower, f_upper = objective(phi_lower, avg_c_bf, new_EV), objective(phi_upper, avg_c_bf, new_EV)
        print(f"Expanding range: phi_lower={phi_lower}, phi_upper={phi_upper}")

        if abs(phi_lower) > 1e5 or abs(phi_upper) > 1e5:  # Prevent infinite expansion
            raise ValueError("Could not find a valid bracket for root-finding.")
    
    # Find root
    result = root_scalar(objective, bracket=[phi_lower, phi_upper], args=(avg_c_bf, new_EV), method=the_method)

    if result.converged:
        if do_print:
            print(f'Consumption at every age before the policy change must change with {round(result.root*100,1)} pct. to keep the same utility')
        return result.root
    else:
        raise ValueError("Root-finding for phi did not converge")