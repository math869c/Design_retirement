from numba import njit, prange
import numpy as np 

from consav.linear_interp import interp_1d, interp_2d, interp_3d, interp_4d
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
    '''Wage before taxes'''
    return par.full_time_hours*np.exp(par.beta_1*k + par.beta_2*t**2)

# 1.1 The four sources of income all before taxes and retirement contributions - and total income before taxes and retirement contributions:
# 1.1.1 Capital income
@jit_if_enabled(fastmath=True)
def capital_return_fct(par, a):
    '''Capital return is the same for all periods'''
    return (par.r_a/(1+par.r_a)) * a

# 1.1.2. Retirement payouts
@jit_if_enabled(fastmath=True)
def calculate_retirement_payouts(par, s, r, t):
    """Calculate retirement payouts: can be split into 3 periods: before retirement, during installment and annuity, and only annuity"""
    if t >= r + par.m:
        EL = round(sum(np.cumprod(par.pi_el[int(r):])*np.arange(int(r),par.T))/(par.T-int(r)),0)
        s_retirement = s
        s_lr =  (((1+par.r_s)**EL)*s_retirement*par.share_lr)/np.sum((1+par.r_s)**(np.arange(EL)))
        return  s_lr, 0.0
    
    elif t >= r:
        EL = round(sum(np.cumprod(par.pi_el[int(r):])*np.arange(int(r),par.T))/(par.T-int(r)),0)
        s_retirement = s
        s_lr =  (((1+par.r_s)**EL)*s_retirement*par.share_lr)/np.sum((1+par.r_s)**(np.arange(EL)))
        s_rp = (((1+par.r_s)**par.m)*s_retirement*(1-par.share_lr))/np.sum((1+par.r_s)**(np.arange(par.m)))
        return   s_lr, s_rp
    else:
        return 0.0, 0.0

# 1.1.3. Labor income
@jit_if_enabled(fastmath=True)
def labor_income_fct(par, k, h, r, t):
    '''Before and after retirement age'''
    return h*wage(par, k, t)

# 1.1.4. Public benefits
@jit_if_enabled(fastmath=True)
def public_benefit_fct(par, h, e, income, t):
    """Before retirement: unemployment benefits (if working, then no benefits), after retirement: public pension"""
    # Before public retirement age
    if t < par.retirement_age:
        if e == 2:
            # Unemployment benefits
            return par.early_benefit
        elif int(h) == 0:
            # Unemployment benefits
            return par.unemployment_benefit
        else:
            # No public benefits
            return 0.0
        
    # public retirement benefits
    else:
        base_payment = par.chi_base
        exceed = np.maximum(0, income - par.chi_max)
        extra_pension = np.maximum(0, par.chi_extra_start - exceed*par.rho)
        retirement_income = base_payment + extra_pension
        return retirement_income

# 1.1.5 Total income before taxes and retirement contributions
@jit_if_enabled(fastmath=True)
def income_private_fct(par, a, s, k, h, e, r, t):
    '''Private income is taxed and is used for 
    Income before taxes and contribution: include capital return, retirement payouts, and wages'''
    # Capital income
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?

    # Retirement payouts
    if t >= r:
        s_lr, s_rp = calculate_retirement_payouts(par, s, r, t)
    else :
        s_lr, s_rp = 0.0, 0.0
    
    # labor income 
    labor_income = labor_income_fct(par, k, h, r, t)

    # Total income 
    total_income = labor_income + a_return + s_lr + s_rp

    # public benefits
    public_benefit = public_benefit_fct(par, h, e, total_income, t)

    return total_income + public_benefit


@jit_if_enabled(fastmath=True)
def tax_rate_fct(par, a, s, k, h, e, r, t):

    total_income = income_private_fct(par, a, s, k, h, e, r, t)

    am_aar = total_income * par.am_sats
    personlignd_aar = (total_income - am_aar)
    grundlag_aar = max(personlignd_aar - par.personfradrag, 0)
    beskfradrag_aar = min(par.beskfradrag_graense, total_income * par.beskfradrag_sats)
    skattepligt_aar = (grundlag_aar - beskfradrag_aar)
    bundskat_aar = (grundlag_aar * par.bundskat_sats)
    topskat_aar = (par.topskat_sats * max(personlignd_aar - par.topskat_graense, 0))
    kommuneskat_aar = (par.kommuneskat_sats * skattepligt_aar)

    # Notice kirkeskat was never added to 'indkomstskat' in the original code
    indkomstskat_aar = (am_aar + bundskat_aar + topskat_aar + kommuneskat_aar)
    ind_efter_aar = (total_income - indkomstskat_aar)

    # Effective tax rate
    skatteprocent_aar = max(1 - (ind_efter_aar / total_income), 0)
    return skatteprocent_aar




# 1.2.2 retirement contributions, only of labor income 
@jit_if_enabled(fastmath=True)
def retirement_contribution_fct(par, a, s, k, h, r, t):
    '''Retirement contributions'''
    if t >= r:
        return 0.0
    else:
        return labor_income_fct(par, k, h, r, t)*par.tau[t]

# 1.3. calculate income after taxes and contributions
# 1.3.1 Income after taxes and contributions
@jit_if_enabled(fastmath=True)
def final_income_and_retirement_contri(par, a, s, k, h, e, r, t):
    ''' The following is taxed: capital income, retirement payouts, labor income, and public benefits
    retirement contributions are only of labor income'''
    # Incomes before taxes and contributions
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?
    if t >= r:
        s_lr, s_rp = calculate_retirement_payouts(par, s, r, t)
    else:
        s_lr, s_rp = 0.0, 0.0
    
    labor_income = labor_income_fct(par, k, h, r, t)
    income_private = a_return + s_lr + s_rp + labor_income
    chi = public_benefit_fct(par, h, e, income_private, t)

    # Tax rate and retirement contribution
    tax_rate = tax_rate_fct(par, a, s, k, h, e, r, t)
    retirement_contribution = retirement_contribution_fct(par, a, s, k, h, r, t)
    if t >= r:
        # income after taxes&contributions
        return (1-tax_rate)*(income_private + chi), 0.0
    else:
        # income after taxes, and retirement contributions 
        return (1-tax_rate)*(income_private*(1-par.tau[t]) + chi), retirement_contribution


# 2. Helper functions in solving and optimizing
@jit_if_enabled(fastmath=True)
def budget_constraint(par, h, a, s, k, e, r, t):
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)
    return par.c_min, max(par.c_min*2, a + income)



@jit_if_enabled(fastmath=True)
def precompute_EV_next(par, sol_V, retirement_idx, employed_idx, t):

    EV = np.zeros((len(par.a_grid), len(par.s_grid), len(par.k_grid)))


    V_next_un       = sol_V[t+1, :, :, :, retirement_idx, 0]
    V_next_em       = sol_V[t+1, :, :, :, retirement_idx, 1]
    V_next_early    = sol_V[t+1, :, :, :, retirement_idx, 2]

    if t < par.first_retirement:
        if employed_idx == 0:
            V_next = (1-par.hire[t] - par.p_early[t])*V_next_un + par.hire[t]*V_next_em + par.p_early[t] * V_next_early
        if employed_idx == 1:
            V_next = par.fire[t]*V_next_un + (1-par.fire[t]-par.p_early[t])*V_next_em + par.p_early[t] * V_next_early
        else: 
            V_next = V_next_early

    elif t < par.retirement_age:
        if t == par.retirement_age - 1:
            if employed_idx == 0:
                V_next  = V_next_un
            elif employed_idx == 1:
                V_next = par.p_early[t]*V_next_un + (1-par.fire[t])*V_next_em 
            else:
                V_next = V_next_un # early retirement no longer exists
        else:
            if employed_idx == 0:
                V_next = (1- par.p_early[t])*V_next_un + par.p_early[t] * V_next_early
            elif employed_idx == 1:
                V_next = par.fire[t]*V_next_un + (1-par.fire[t]-par.p_early[t])*V_next_em + par.p_early[t] * V_next_early
            else:
                V_next = V_next_early
    
    elif t < par.last_retirement:
        if t == par.last_retirement - 1:
            V_next = V_next_un
        else:
            if employed_idx == 0:
                V_next = V_next_un
            elif employed_idx == 1:
                V_next = (1-par.fire[t])*V_next_un + par.fire[t]*V_next_em
            else: 
                V_next = V_next_early

    else:
        V_next = V_next_un

    # V_next = V_next_un

    for i_a, a_next in enumerate(par.a_grid):
        for i_s, s_next in enumerate(par.s_grid):
            for i_k, k_next in enumerate(par.k_grid[t]):

                EV_val = 0.0
                for idx in range(par.N_xi):
                    k_temp_ = k_next*par.xi_v[idx] 
                    V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], V_next, a_next, s_next, k_temp_)
                    EV_val += V_next_interp * par.xi_p[idx]

                EV[i_a, i_s, i_k] = EV_val

    return EV
@jit_if_enabled(fastmath=True)
def calculate_last_period_consumption(par, a, s, e, r, t):
    k, h = 0.0, 0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)
 
    if par.mu != 0.0:
        # With bequest motive
        return max(((1/(1+(par.mu*(1+par.r_a))**(-1/par.sigma)*(1+par.r_a))) 
                    * (par.mu*(1+par.r_a))**(-1/par.sigma) 
                    * ((1+par.r_a)*(a+income)+par.a_bar)), 0)
    
    else: 
        # No bequest motive
        return (a + income)


# 3. Value functions
@jit_if_enabled(fastmath=True)
def value_last_period(par, c, a, s, e, r, t):
    # states and income 
    h, k = 0.0,0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)
    a_next = (1+par.r_a)*(a + income - c)

    return utility(par, c, h) + bequest(par, a_next)

@jit_if_enabled(fastmath=True)
def value_function_after_retirement(par, sol_V, c, a, s, e, r, t):
    # states and income 
    retirement_age_idx = r - par.first_retirement
    e_idx = int(0)
    h, k  = 0.0, 0.0
    k_idx = 0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)

    # Next period states 
    a_next = (1+par.r_a)*(a + income - c)
    s_next = s
    V_next = sol_V[t+1, :, :, k_idx, retirement_age_idx, e_idx]
    EV_next = interp_2d(par.a_grid, par.s_grid, V_next, a_next, s_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, e, t):
    # states and income 
    income, retirement_contribution = final_income_and_retirement_contri(par, a, s, k, h, e, par.last_retirement, t)

    # Next period states
    a_next = (1+par.r_a)*(a + income - c)
    s_next = (1+par.r_s)*(s + retirement_contribution)
    k_next = ((1-par.delta)*k + h)
    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_EV, a_next, s_next, k_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)

# 4. Objective functions 
@jit_if_enabled(fastmath=True)
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, e, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, e, t)

@jit_if_enabled()
def obj_consumption_after_retirement(c, par, sol_V, a, s, e, r, t):
    return -value_function_after_retirement(par, sol_V, c, a, s, e, r, t)

@jit_if_enabled(fastmath=True)
def obj_hours(h, par, sol_V, sol_EV, a, s, k, e, r, t, dist):

    bc_min, bc_max = budget_constraint(par, h, a, s, k, e, r, t)
    
    c_star = optimizer(
        obj_consumption,     
        bc_min, 
        bc_max,
        args=(par, sol_V, sol_EV, h, a, s, k, e, t),
        tol=dist
    )
    
    val_at_c_star = -value_function(par, sol_V, sol_EV, c_star, h, a, s, k, e, t)
    
    return val_at_c_star

# 5. Solving the model
@jit_if_enabled(parallel=True)
def main_solver_loop(par, sol, do_print = False):

    human_capital_unemp, hours_unemp, e_unemployed = 0.0, 0.0, 0.0

    sol_a = sol.a
    sol_ex = sol.ex
    sol_c = sol.c
    sol_h = sol.h
    sol_V = sol.V

    
    for t in range(par.T - 1, -1, -1):
        if do_print:
            print(f"We are in t = {t}")

        count = 0

        retirement_ages = np.arange(par.first_retirement, min(par.last_retirement + 1, t + 1)) \
                        if t >= par.first_retirement + 1 else np.arange(par.first_retirement, par.first_retirement + 1)

        for retirement_age_idx, retirement_age in enumerate(retirement_ages):

            if t > par.last_retirement:
                e_grid = [0]
            elif t >= par.retirement_age:
                e_grid = [0, 1]
            else:
                e_grid = [0, 1, 2]

            for employed_idx in e_grid:
                employed = employed_idx

                if t <= retirement_age:
                    sol_EV = precompute_EV_next(par, sol_V, retirement_age_idx, employed_idx, t)

                for a_idx in prange(len(par.a_grid)):
                    assets = par.a_grid[a_idx]

                    for s_idx in range(len(par.s_grid)):
                        savings = par.s_grid[s_idx]

                        for k_idx in range(len(par.k_grid[t])):
                            human_capital = par.k_grid[t][k_idx]

                            idx = (t, a_idx, s_idx, k_idx, retirement_age_idx, employed_idx)
                            idx_unemployed = (t, a_idx, s_idx, k_idx, retirement_age_idx, int(e_unemployed))
                            idx_next = (t + 1, a_idx, s_idx, k_idx, retirement_age_idx, employed_idx)

                            if t == par.T - 1: # Last period
                                income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, t)
                                cash_on_hand = assets + income

                                sol_c[idx] = calculate_last_period_consumption(par, assets, savings, employed, retirement_age, t)
                                sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                sol_ex[idx] = employed
                                sol_h[idx] = hours_unemp
                                sol_V[idx] = value_last_period(par, sol_c[idx], assets, savings, employed, retirement_age, t)

                                count += 1

                            elif t > retirement_age: # After retirement age, with "ratepension"
                                if employed == int(0.0):

                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, t)

                                    c_star = optimizer(
                                        obj_consumption_after_retirement,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, assets, savings, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, t)
                                    cash_on_hand = assets + income


                                    sol_c[idx] = c_star
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                    sol_ex[idx] = employed
                                    sol_h[idx] = hours_unemp
                                    sol_V[idx] = value_function_after_retirement(par, sol_V, c_star, assets, savings, employed, retirement_age, t)

                                    count += 1

                                else:

                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, t)

                                    c_star = optimizer(
                                        obj_consumption_after_retirement,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, assets, savings, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, t)
                                    cash_on_hand = assets + income


                                    sol_c[idx] = c_star
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                    sol_ex[idx] = e_unemployed
                                    sol_h[idx] = hours_unemp
                                    sol_V[idx] = value_function_after_retirement(par, sol_V, c_star, assets, savings, employed, retirement_age, t)

                                    count += 1                                

                            elif t == retirement_age and sol_ex[idx_next] == 0.0:

                                if employed == int(0.0): # Forced unemployment
                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, retirement_age, t)

                                    c_star_u = optimizer(
                                        obj_consumption_after_retirement,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, assets, savings, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, retirement_age, t)
                                    cash_on_hand_un = assets + income


                                    sol_V[idx] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, t)
                                    sol_c[idx]  = c_star_u
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                    sol_ex[idx] = e_unemployed
                                    sol_h[idx]  = hours_unemp

                                    count += 1

                                elif employed == int(1.0): # Can choose between employment and unemployment
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, par.last_retirement, t),
                                        tol=par.opt_tol
                                    )

                                    bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, par.last_retirement, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, t),
                                        tol=par.opt_tol
                                    )
                                    val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, t)
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, par.last_retirement, t)
                                    cash_on_hand = assets + income


                                    if sol_V[idx_unemployed] > val:
                                        sol_V[idx] = sol_V[idx_unemployed]
                                        sol_c[idx] = sol_c[idx_unemployed]
                                        sol_a[idx] = sol_a[idx_unemployed]
                                        sol_ex[idx] = e_unemployed
                                        sol_h[idx]  = sol_h[idx_unemployed]
                                    else:
                                        sol_V[idx] = val
                                        sol_ex[idx] = employed
                                        sol_h[idx]  = h_star
                                        sol_c[idx] = c_star
                                        sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                else: # Forced unemployment
                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, retirement_age, t)

                                    c_star_u = optimizer(
                                        obj_consumption_after_retirement,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, assets, savings, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, retirement_age, t)
                                    cash_on_hand_un = assets + income


                                    sol_V[idx] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, t)
                                    sol_c[idx]  = c_star_u
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                    sol_ex[idx] = e_unemployed
                                    sol_h[idx]  = hours_unemp

                                    count += 1

                            else:
                                if employed == int(0.0): # Forced unemployment
                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, par.last_retirement, t)
                                    
                                    c_star_u = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, hours_unemp, assets, savings, human_capital, employed, t),
                                        tol=par.opt_tol
                                    )
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, par.last_retirement, t)
                                    cash_on_hand_un = assets + income

                                    sol_V[idx] = value_function(par, sol_V, sol_EV, c_star_u, hours_unemp, assets, savings, human_capital, employed, t)
                                    sol_c[idx]  = c_star_u
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                    sol_ex[idx] = e_unemployed
                                    sol_h[idx]  = hours_unemp

                                if employed == int(1.0): # Can choose between employment and unemployment
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, par.last_retirement, t),
                                        tol=par.opt_tol
                                    )

                                    bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, par.last_retirement, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, t),
                                        tol=par.opt_tol
                                    )

                                    val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, t)
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, par.last_retirement, t)
                                    cash_on_hand = assets + income
                                    
                                    if sol_V[idx_unemployed] > val:
                                        sol_V[idx] = sol_V[idx_unemployed]
                                        sol_c[idx] = sol_c[idx_unemployed]
                                        sol_ex[idx] = e_unemployed
                                        sol_h[idx]  = sol_h[idx_unemployed] 
                                        sol_a[idx] = sol_a[idx_unemployed]
                                    
                                    else:
                                        sol_V[idx] = val
                                        sol_c[idx] = c_star
                                        sol_ex[idx] = employed
                                        sol_h[idx] = h_star
                                        sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])

                                else: # Forced unemployment
                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, par.last_retirement, t)
                                    
                                    c_star_u = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, hours_unemp, assets, savings, human_capital, employed, t),
                                        tol=par.opt_tol
                                    )
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, par.last_retirement, t)
                                    cash_on_hand_un = assets + income

                                    sol_V[idx] = value_function(par, sol_V, sol_EV, c_star_u, hours_unemp, assets, savings, human_capital, employed, t)
                                    sol_c[idx]  = c_star_u
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                    sol_ex[idx] = e_unemployed
                                    sol_h[idx]  = hours_unemp


    return sol_c, sol_h, sol_ex, sol_V, sol_a

# 6. simulation:
@jit_if_enabled(parallel=True)
def main_simulation_loop(par, sol, sim, do_print = False):
    '''Simulate the model: structure within each periode:
        1. technical variables
        2. interpolation of optimal consumption and hours
        3. income variables
        4. update of states'''
    sim_a = sim.a
    sim_s = sim.s
    sim_k = sim.k
    sim_c = sim.c
    sim_h = sim.h
    sim_e = sim.e

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
    sim_tax_rate = sim.tax_rate
    sim_income_before_tax_contrib = sim.income_before_tax_contrib
    sim_e_init = sim.e_init
    sim_from_employed = sim.from_employed
    sim_from_unemployed = sim.from_unemployed
    sim_from_unemployed_to_only_early = sim.from_unemployed_to_only_early
    sim_from_employed_to_unemployed = sim.from_employed_to_unemployed
    
    sim_s_retirement_contrib = sim.s_retirement_contrib
    
    sol_ex = sol.ex
    sol_c = sol.c
    sol_h = sol.h

    # i. initialize states
    sim_a[:,0] = sim_a_init[:]
    sim_s[:,0] = sim_s_init[:]
    sim_k[:,0] = sim_k_init[:]
    sim_e[:,0] = sim_e_init[:]

    for t in range(par.simT):

        # ii. interpolate optimal consumption and hours
        if t < par.first_retirement:
            for i in prange(par.simN):
                if t == 0:
                    pass
                else:
                    if sim_e[i,t-1] == 2.0:
                        sim_e[i,t] = 2.0
                    elif sim_ex[i,t-1] == 1.0:
                        sim_e[i,t] = sim_from_employed[i,t]
                    else:
                        sim_e[i,t] = sim_from_unemployed[i,t]
                
                # 1. technical variables
                retirement_age_idx[i] = np.minimum(np.maximum(t-par.first_retirement, 0), par.last_retirement-par.first_retirement)

                if sim_e[i,t] == 1.0:
                    sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_ex[i,t] = np.round(sim_ex[i,t])
                else:
                    sim_ex[i,t] = 0.0

                if sim_ex[i,t] == 0.0:
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age_idx[i]), int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = 0.0

                if sim_ex[i,t] == 1.0:
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age_idx[i]), int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age_idx[i]), int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])

                # 3. Income variables 
                # 3.1 final income and retirement payments 
                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t)
                # 3.2 labor income
                sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                # 3.3 public benefits
                sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                # 3.4 income before tax contribution
                sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t) 
                # 3.5 tax rate
                sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t)

                # 4. Update of states
                sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                sim_s[i,t+1] = np.minimum((1+par.r_s)*(sim_s[i,t] + sim_s_retirement_contrib[i,t]), par.s_max)
                sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])

        elif t < par.retirement_age:
            for i in prange(par.simN):

                if sim_e[i,t-1] == 2.0:
                    sim_e[i,t] = 2.0
                    sim_ex[i,t] = 0.0
                elif sim_ex[i,t-1] == 1.0:
                    sim_e[i,t] = sim_from_employed[i,t]
                    if sim_e[i,t] == 1.0:
                        retirement_age_idx[i] = np.minimum(np.maximum(t-par.first_retirement, 0), par.last_retirement-par.first_retirement)
                        sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                        sim_ex[i,t] = np.round(sim_ex[i,t])
                    else: 
                        sim_ex[i,t] = 0.0
                else: # just unemployed
                    if sim_from_unemployed_to_only_early[i,t] == 1.0:
                        sim_e[i,t] = 2.0
                        sim_ex[i,t] = 0.0
                    else:
                        sim_e[i,t] = 0.0
                        sim_ex[i,t] = 0.0


                # 1. technical variables
                if sim_ex[i,t] == 0.0:
                    if (sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 1.0) or (sim_ex[i,t-1] == 1.0 and t == par.last_retirement) or (t == par.first_retirement and sim_ex[i,t] == 0.0): 
                        # 1.1 retirement age
                        retirement_age[i] = t
                        retirement_age_idx[i] = np.minimum(np.maximum(t-par.first_retirement, 0), par.last_retirement-par.first_retirement)
                        s_retirement[i] = sim_s[i,t]

                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = 0.0

                    elif sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 0.0: 
                        # 1.1 retirement age

                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                        sim_h[i,t] = 0.0

                        # 3. Income variables
                    sim_income[i,t], _ = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)
                    # 3.1 retirement payments 
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, s_retirement[i], retirement_age[i], t)
                    # 3.2 labor income 
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    # 4. Update of states
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])

                else: 
                    # 1.1 retirement age
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,t-par.first_retirement, int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,t-par.first_retirement, int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])

                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t)
                    # 3.1 retirement payments
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t)

                    # 4. Update of states
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum((1+par.r_s)*(sim_s[i,t] + sim_s_retirement_contrib[i,t]), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])





        elif t <= par.last_retirement:
            for i in prange(par.simN):

                if sim_e[i,t-1] == 2.0: #Førtidspension eksisterere ikke længere og man skal overgå til pension
                    sim_e[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                elif sim_ex[i,t-1] == 1.0:
                    if sim_from_employed_to_unemployed[i,t] == 1.0:
                        sim_e[i,t] = 0.0
                        sim_ex[i,t] = 0.0
                    else:
                        sim_e[i,t] = 1.0
                        retirement_age_idx[i] = np.minimum(np.maximum(t-par.first_retirement, 0), par.last_retirement-par.first_retirement)
                        sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                        sim_ex[i,t] = np.round(sim_ex[i,t])
                else: # just unemployed
                    sim_e[i,t] = 0.0
                    sim_ex[i,t] = 0.0


                # 1. technical variables

                if sim_ex[i,t] == 0.0 or t == par.last_retirement:
                    if (sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 1.0) or (sim_ex[i,t-1] == 1.0 and t == par.last_retirement) or (t == par.first_retirement and sim_ex[i,t] == 0.0): 
                        # 1.1 retirement age
                        retirement_age[i] = t
                        retirement_age_idx[i] = np.minimum(np.maximum(t-par.first_retirement, 0), par.last_retirement-par.first_retirement)
                        s_retirement[i] = sim_s[i,t]

                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = 0.0

                    elif sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 0.0: 
                        # 1.1 retirement age

                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                        sim_h[i,t] = 0.0

                        # 3. Income variables
                    sim_income[i,t], _ = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)
                    # 3.1 retirement payments 
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, s_retirement[i], retirement_age[i], t)
                    # 3.2 labor income 
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    # 4. Update of states
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])



                else: 
                    # 1.1 retirement age
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,t-par.first_retirement, int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,t-par.first_retirement, int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])

                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t)
                    # 3.1 retirement payments
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], sim_e[i,t], par.last_retirement, t)

                    # 4. Update of states
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum((1+par.r_s)*(sim_s[i,t] + sim_s_retirement_contrib[i,t]), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])

        elif t > par.last_retirement:
            sim_ex[:,t] = 0.0
            sim_e[:,t]  = 0.0

            for i in prange(par.simN):
                # 1.1 retirement age

                # 2. Interpolation of choice variables
                sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                sim_h[i,t] = 0.0

                # 3. Income variables
                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)
                # 3.1 retirement payments
                # 3.2 labor income
                sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                # 3.3 public benefits
                sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                # 3.4 income before tax contribution
                sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                # 3.5 tax rate
                sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                if t < retirement_age[i] + par.m: 
                    # 4. Update of states
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])

                elif par.T - 1 > t >= retirement_age[i] + par.m:
                    # 4. Update of states
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] - sim_s_lr_init[i])*(1+par.r_s),0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])
                   

    return sim_a, sim_s, sim_k, sim_c, sim_h, sim_w, sim_ex, sim_e, sim_chi_payment, sim_tax_rate, sim_income_before_tax_contrib, s_retirement, retirement_age, sim_income
