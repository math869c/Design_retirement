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
@jit_if_enabled(fastmath=False)
def utility(par, c, h, k):
    return ((c+1)**(1-par.sigma))/(1-par.sigma) - (par.zeta/(1+k)) * (h**(1+par.gamma))/(1+par.gamma)

@jit_if_enabled(fastmath=False)
def bequest(par, a):
    return par.mu*(a+par.a_bar)**(1-par.sigma) / (1-par.sigma)

@jit_if_enabled(fastmath=False)
def wage(par, k, t):
    '''Wage before taxes'''
    return par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)

# 1.1 The four sources of income all before taxes and retirement contributions - and total income before taxes and retirement contributions:
# 1.1.1 Capital income
@jit_if_enabled(fastmath=False)
def capital_return_fct(par, a):
    '''Capital return is the same for all periods'''
    return (par.r_a/(1+par.r_a)) * a

# 1.1.2. Retirement payouts
@jit_if_enabled(fastmath=False)
def calculate_retirement_payouts(par, h, s, e, r, t):
    """Calculate retirement payouts: can be split into 3 periods: before retirement, during installment and annuity, and only annuity"""
    
    if (e == 2) or (t >= par.first_retirement and h == 0.0):
        if t >= r + par.m:
            s_retirement = s
            s_lr =  (((1+par.r_s)**par.EL[int(r)])*s_retirement*par.share_lr)/np.sum((1+par.r_s)**(np.arange(par.EL[int(r)])))
            return  s_lr, 0.0
        
        elif t >= r:
            s_retirement = s
            s_lr =  (((1+par.r_s)**par.EL[int(r)])*s_retirement*par.share_lr)/np.sum((1+par.r_s)**(np.arange(par.EL[int(r)])))
            s_rp = (((1+par.r_s)**par.m)*s_retirement*(1-par.share_lr))/np.sum((1+par.r_s)**(np.arange(par.m)))
            return   s_lr, s_rp
        
        else:
            print(e, t, r)
            print("Error: Invalid retirement payout calculation")
    else:
        return 0.0, 0.0

# 1.1.3. Labor income
@jit_if_enabled(fastmath=False)
def labor_income_fct(par, k, h, r, t):
    '''Before and after retirement age'''
    return h*wage(par, k, t)

# 1.1.4. Public benefits
@jit_if_enabled(fastmath=False)
def public_benefit_fct(par, h, e, income, t):
    """Before retirement: unemployment benefits (if working, then no benefits), after retirement: public pension"""
    # Before public retirement age
    if t < par.retirement_age:
        if h > par.h_min:
            return 0.0
        elif e == par.emp or e == par.unemp:
            # Unemployment benefits
            return par.unemployment_benefit[t]
        elif e == par.ret:
            # Retirement benefits
            return par.early_benefit[t]
        else:
            print("Error: Invalid employment status")
            return par.unemployment_benefit[t]
        
    # public retirement benefits
    else:
        return max(par.chi_base, par.chi_total - income*par.rho)

# 1.1.5 Total income before taxes and retirement contributions
@jit_if_enabled(fastmath=False)
def income_private_fct(par, a, s, k, h, e, r, t):
    '''Private income is taxed and is used for 
    Income before taxes and contribution: include capital return, retirement payouts, and wages'''
    # Capital income
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?

    # Retirement payouts
    if (e == 2) or (t >= par.first_retirement and h == 0.0):
        s_lr, s_rp = calculate_retirement_payouts(par, h, s, e, r, t)  # par, h, s, e, r, t
    else:
        s_lr, s_rp = 0.0, 0.0
    
    # labor income 
    labor_income = labor_income_fct(par, k, h, r, t)

    # Total income 
    total_income = labor_income + a_return + s_lr + s_rp

    # public benefits
    public_benefit = public_benefit_fct(par, h, e, total_income, t)

    return total_income + public_benefit


@jit_if_enabled(fastmath=False)
def tax_rate_fct(par, a, s, k, h, e, r, t):
    total_income = income_private_fct(par, a, s, k, h, e, r, t)

    labor_market_contribution = total_income * par.labor_market_rate
    personal_income = total_income - labor_market_contribution
    taxable_base = max(personal_income - par.personal_allowance, 0)
    employment_deduction = min(par.employment_deduction_cap, total_income * par.employment_deduction_rate)
    taxable_income = taxable_base - employment_deduction
    bottom_tax = taxable_base * par.bottom_tax_rate
    top_tax = par.top_tax_rate * max(personal_income - par.top_tax_threshold, 0)
    municipal_tax = par.municipal_tax_rate * taxable_income

    # Note: Church tax is not included in 'income_tax'
    income_tax = labor_market_contribution + bottom_tax + top_tax + municipal_tax
    income_after_tax = total_income - income_tax

    # Effective tax rate
    effective_tax_rate = max(1 - (income_after_tax / total_income), 0)
    return effective_tax_rate
    # return par.upsilon


# 1.2.2 retirement contributions, only of labor income 
@jit_if_enabled(fastmath=False)
def retirement_contribution_fct(par, a, s, k, h, r, t):
    '''Retirement contributions'''
    return labor_income_fct(par, k, h, r, t)*par.tau[t]


# 1.3. calculate income after taxes and contributions
# 1.3.1 Income after taxes and contributions
@jit_if_enabled(fastmath=False)
def final_income_and_retirement_contri(par, a, s, k, h, e, r, t):
    ''' The following is taxed: capital income, retirement payouts, labor income, and public benefits
    retirement contributions are only of labor income'''
    # Incomes before taxes and contributions
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?
    if (e == 2) or (t >= par.first_retirement and h == 0.0):
        s_lr, s_rp = calculate_retirement_payouts(par, h, s, e, r, t)
    else:
        s_lr, s_rp = 0.0, 0.0
    
    labor_income = labor_income_fct(par, k, h, r, t)
    income_private = a_return + s_lr + s_rp + labor_income
    chi = public_benefit_fct(par, h, e, income_private, t)

    # Tax rate and retirement contribution
    tax_rate = tax_rate_fct(par, a, s, k, h, e, r, t)
    retirement_contribution = retirement_contribution_fct(par, a, s, k, h, r, t)

    if h > 0.0:
        return (1-tax_rate)*(income_private*(1-par.tau[t]) + chi), retirement_contribution
    else:
        return (1-tax_rate)*(income_private + chi), retirement_contribution


# 2. Helper functions in solving and optimizing
@jit_if_enabled(fastmath=False)
def budget_constraint(par, h, a, s, k, e, r, t):
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)
    return par.c_min, max(par.c_min*2, a + income)


@jit_if_enabled(fastmath=False)
def compute_transitions(par, sol_V, employed, retirement_idx, ex_next, t):
    
    if t == par.last_retirement:
        V_next_em       = sol_V[t+1, :, :, :, retirement_idx, par.ret]
        V_next_un       = sol_V[t+1, :, :, :, retirement_idx, par.ret]
        V_next_early    = sol_V[t+1, :, :, :, retirement_idx, par.ret]

    elif t >= par.retirement_age - 1:
        if int(ex_next) == par.unemp:
            V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]
        else:
            V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, par.emp]

        V_next_un       = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]
        V_next_early    = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]

    elif t >= par.first_retirement - 1:
        if int(ex_next) == par.unemp:
            V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, par.unemp]
        else:
            V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, par.emp]

        V_next_un       = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]
        V_next_early    = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]

    else:
        V_next_em       = sol_V[t+1, :, :, :, retirement_idx+1, int(ex_next)]
        V_next_un       = sol_V[t+1, :, :, :, retirement_idx+1, par.unemp]
        V_next_early    = sol_V[t+1, :, :, :, retirement_idx+1, par.ret]

    V_next = par.p_e_0[t]*V_next_un + par.p_e_1[t]*V_next_em + par.p_e_2[t] * V_next_early

    return V_next


@jit_if_enabled(fastmath=False)
def precompute_EV_next(par, sol_ex, sol_V, retirement_idx, employed, t):

    EV = np.zeros((len(par.a_grid), len(par.s_grid), len(par.k_grid[t])))

    for i_a, a_next in enumerate(par.a_grid):
        for i_s, s_next in enumerate(par.s_grid):
            for i_k, k_next in enumerate(par.k_grid[t]):
                EV_val = 0.0
                for idx in range(par.N_xi):
                    k_temp_ = k_next*par.xi_v[idx] 

                    if t == par.last_retirement:
                        ex_next = 0


                    
                    else:
                        if employed == par.emp:
                            ex_next = np.round(interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t+1, :, :, :, retirement_idx+1, employed], a_next, s_next, k_temp_))
                        else:
                            ex_next = par.unemp

                    V_next = compute_transitions(par, sol_V, employed, retirement_idx, ex_next, t)
                    V_next_interp = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], V_next, a_next, s_next, k_temp_)
                    EV_val += V_next_interp * par.xi_p[idx]

                EV[i_a, i_s, i_k] = EV_val

    return EV


@jit_if_enabled(fastmath=False)
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
@jit_if_enabled(fastmath=False)
def value_last_period(par, c, a, s, e, r, t):
    # states and income 
    h, k = 0.0,0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)
    a_next = (1+par.r_a)*(a + income - c)

    return utility(par, c, h, k) + bequest(par, a_next)


@jit_if_enabled(fastmath=False)
def value_function_after_retirement(par, sol_V, c, a, s, e, r, t):
    # states and income 
    retirement_age_idx = r
    e_idx = par.ret
    h, k  = 0.0, 0.0
    k_idx = 0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)

    # Next period states 
    a_next = (1+par.r_a)*(a + income - c)
    s_next = s
    V_next = sol_V[t+1, :, :, k_idx, retirement_age_idx, e_idx]
    EV_next = interp_2d(par.a_grid, par.s_grid, V_next, a_next, s_next)

    return utility(par, c, h, k) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=False)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, e, r, t):
    # states and income 
    income, retirement_contribution = final_income_and_retirement_contri(par, a, s, k, h, e, r, t)

    # Next period states
    a_next = (1+par.r_a)*(a + income - c)
    s_next = (1+par.r_s)*(s + retirement_contribution)
    k_next = ((1-par.delta)*k + h)
    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_EV, a_next, s_next, k_next)

    return utility(par, c, h, k) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


# 4. Objective functions 
@jit_if_enabled(fastmath=False)
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, e, r, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, e, r, t)


@jit_if_enabled()
def obj_consumption_after_retirement(c, par, sol_V, a, s, e, r, t):
    return -value_function_after_retirement(par, sol_V, c, a, s, e, r, t)


@jit_if_enabled(fastmath=False)
def obj_hours(h, par, sol_V, sol_EV, a, s, k, e, r, t, dist):

    bc_min, bc_max = budget_constraint(par, h, a, s, k, e, r, t)
    
    c_star = optimizer(
        obj_consumption,     
        bc_min, 
        bc_max,
        args=(par, sol_V, sol_EV, h, a, s, k, e, r, t),
        tol=dist
    )
    
    val_at_c_star = -value_function(par, sol_V, sol_EV, c_star, h, a, s, k, e, r, t)
    
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

        retirement_ages = np.arange(0, min(par.last_retirement + 1, t + 1))

        for retirement_age_idx, retirement_age in enumerate(retirement_ages):

            if t > par.last_retirement:
                e_grid = [par.ret]
            elif t >= par.retirement_age:
                e_grid = [par.emp, par.ret]
            else:
                e_grid = [par.unemp, par.emp, par.ret]

            for employed in e_grid:

                if t <= retirement_age:
                    sol_EV = precompute_EV_next(par, sol_ex, sol_V, retirement_age_idx, employed, t)

                for a_idx in prange(len(par.a_grid)):
                    assets = par.a_grid[a_idx]

                    for s_idx in range(len(par.s_grid)):
                        savings = par.s_grid[s_idx]

                        for k_idx in range(len(par.k_grid[t])):
                            human_capital = par.k_grid[t][k_idx]

                            idx = (t, a_idx, s_idx, k_idx, retirement_age_idx, employed)
                            idx_unemployed = (t, a_idx, s_idx, k_idx, retirement_age_idx, par.unemp)
                            idx_ret = (t, a_idx, s_idx, slice(None), retirement_age_idx, par.ret)

                            if t == par.T - 1: # Last period
                                if k_idx == 0: # No capital
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, t)
                                    cash_on_hand = assets + income

                                    sol_c[idx_ret] = calculate_last_period_consumption(par, assets, savings, employed, retirement_age, t)
                                    sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                    sol_ex[idx_ret] = e_unemployed
                                    sol_h[idx_ret] = hours_unemp
                                    sol_V[idx_ret] = value_last_period(par, sol_c[idx], assets, savings, employed, retirement_age, t)

                                else:
                                    pass

                            elif t > retirement_age: # After retirement age, with "ratepension"
                                if k_idx == 0: # No capital
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


                                    sol_c[idx_ret] = c_star
                                    sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand - sol_c[idx])
                                    sol_ex[idx_ret] = e_unemployed
                                    sol_h[idx_ret] = hours_unemp
                                    sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star, assets, savings, employed, retirement_age, t)

                                else:
                                    pass

                            elif t == retirement_age and t >= par.first_retirement:
                                if employed == par.unemp: # Forced unemployment
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

                                elif employed == par.emp: # Can choose between employment and unemployment
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )

                                    bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, retirement_age, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )
                                    val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, retirement_age, t)
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, retirement_age, t)
                                    cash_on_hand = assets + income
                                    sol_V[idx] = val
                                    sol_h[idx]  = h_star
                                    sol_c[idx] = c_star
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])

                                    if sol_V[idx_unemployed] > val:
                                        sol_ex[idx] = e_unemployed
                                    else:
                                        sol_ex[idx] = employed

                                else: # Forced unemployment
                                    if k_idx == 0: # No capital
                                        bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, t)

                                        c_star_u = optimizer(
                                            obj_consumption_after_retirement,
                                            bc_min,
                                            bc_max,
                                            args=(par, sol_V, assets, savings, employed, retirement_age, t),
                                            tol=par.opt_tol
                                        )

                                        income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, t)
                                        cash_on_hand_un = assets + income

                                        sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, t)
                                        sol_c[idx_ret]  = c_star_u
                                        sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                        sol_ex[idx_ret] = e_unemployed
                                        sol_h[idx_ret]  = hours_unemp

                                    else:
                                        pass

                            else:
                                if employed == par.unemp: # Forced unemployment
                                    bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital, employed, retirement_age, t)
                                    
                                    c_star_u = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, hours_unemp, assets, savings, human_capital, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )

                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, hours_unemp, employed, retirement_age, t)
                                    cash_on_hand_un = assets + income

                                    sol_V[idx] = value_function(par, sol_V, sol_EV, c_star_u, hours_unemp, assets, savings, human_capital, employed, retirement_age, t)
                                    sol_c[idx]  = c_star_u
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                    sol_ex[idx] = e_unemployed
                                    sol_h[idx]  = hours_unemp

                                elif employed == par.emp: # Can choose between employment and unemployment
                                    h_star = optimize_outer(
                                        obj_hours,       
                                        par.h_min,
                                        par.h_max,
                                        args=(par, sol_V, sol_EV, assets, savings, human_capital, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )

                                    bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, employed, retirement_age, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, employed, retirement_age, t),
                                        tol=par.opt_tol
                                    )

                                    val = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, employed, retirement_age, t)
                                    income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital, h_star, employed, retirement_age, t)
                                    cash_on_hand = assets + income
                                    sol_V[idx] = val
                                    sol_c[idx] = c_star
                                    sol_h[idx] = h_star
                                    sol_a[idx] = (1+par.r_a)*(cash_on_hand - sol_c[idx])

                                    if sol_V[idx_unemployed] > val:
                                        sol_ex[idx] = e_unemployed
                                    else:
                                        sol_ex[idx] = employed
                                    

                                else: # Forced unemployment
                                    if k_idx == 0: # No capital
                                        bc_min, bc_max = budget_constraint(par, hours_unemp, assets, savings, human_capital_unemp, employed, retirement_age, t)

                                        c_star_u = optimizer(
                                            obj_consumption_after_retirement,
                                            bc_min,
                                            bc_max,
                                            args=(par, sol_V, assets, savings, employed, retirement_age, t),
                                            tol=par.opt_tol
                                        )

                                        income, _ = final_income_and_retirement_contri(par, assets, savings, human_capital_unemp, hours_unemp, employed, retirement_age, t)
                                        cash_on_hand_un = assets + income

                                        sol_V[idx_ret] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, employed, retirement_age, t)
                                        sol_c[idx_ret]  = c_star_u
                                        sol_a[idx_ret] = (1+par.r_a)*(cash_on_hand_un - sol_c[idx])
                                        sol_ex[idx_ret] = e_unemployed
                                        sol_h[idx_ret]  = hours_unemp

                                    else:
                                        pass


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
    sim_ret_flag    = sim.ret_flag
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
    sim_e_exogenous = sim.e_state_exogenous
    # sim_from_employed = sim.from_employed
    # sim_from_unemployed = sim.from_unemployed
    # sim_from_unemployed_to_only_early = sim.from_unemployed_to_only_early
    # sim_from_employed_to_unemployed = sim.from_employed_to_unemployed
    
    sim_s_retirement_contrib = sim.s_retirement_contrib
    
    sol_ex = sol.ex
    sol_c = sol.c
    sol_h = sol.h

    # i. initialize states
    sim_a[:,0] = sim_a_init[:]
    sim_s[:,0] = sim_s_init[:]
    sim_k[:,0] = sim_k_init[:]
    sim_e[:,0] = sim_e_init[:]


    for i in prange(par.simN):
        for t in range(par.simT):
            # ii. interpolate optimal consumption and hours
            if t < par.first_retirement:
            
                if t == 0:
                    retirement_age[i] = t
                    s_retirement[i] = sim_s[i,t]
                else:
                    if sim_e[i,t-1] == 2.0:
                        sim_e[i,t] = 2.0

                    else:
                        sim_e[i,t] = sim_e_exogenous[i,t]

                    if (sim_e[i,t] == 2.0 and sim_e[i,t-1] != 2) or sim_e[i,t] != 2.0:                    
                        retirement_age[i] = t
                        s_retirement[i] = sim_s[i,t] 


                if sim_e[i,t] == 2.0:
                    sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                elif sim_e[i,t] == 1.0:
                    sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t,:,:,:,int(retirement_age[i]),int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_ex[i,t] = np.round(sim_ex[i,t])

                    if sim_ex[i,t] == 1.0:
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_ret_flag[i,:] = 0.0 # hvis de kommer i arbejde igen, så skal de ikke have retirement flag

                    else:
                        sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_h[i,t] = 0.0
                        sim_ret_flag[i,:] = 0.0 # glem alle tidligere
                        sim_ret_flag[i,t] = 1.0

                else:
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = 0.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                # 3. Income variables 
                # 3.1 final income and retirement payments 
                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) #(par, a, s, k, h, e, r, t)
                sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t

                if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                    sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min


                # 3.2 labor income
                sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                # 3.3 public benefits
                sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                # 3.4 income before tax contribution
                sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                # 3.5 tax rate
                sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                # 4. Update of states
                # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])

                # if sim_a[i,t+1] < par.a_min:
                #     print("id", i, "time", t, "asspre", sim_a[i,t], "ass", sim_a[i,t+1], "inc", sim_income[i,t], "c", sim_c[i,t], "ex", sim_ex[i,t], "h", sim_h[i,t], "e", sim_e[i,t], "r", retirement_age[i])

            elif t < par.retirement_age:

                if sim_e[i,t-1] == 2.0:
                    sim_e[i,t] = 2.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                elif sim_ex[i,t-1] == 1.0:
                    sim_e[i,t] = sim_e_exogenous[i,t]
                    retirement_age[i] = t
                    s_retirement[i] = sim_s[i,t]
                    if sim_e[i,t] == 1.0:
                        sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t,:,:,:,int(retirement_age[i]),int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_ex[i,t] = np.round(sim_ex[i,t])
                        sim_e[i,t] = sim_ex[i,t]
                        if sim_ex[i,t] == 0.0:
                            sim_ret_flag[i,t] = 1.0
                    else: 
                        sim_ex[i,t] = 0.0
                        sim_ret_flag[i,t] = 0.0

                else: # just unemployed
                    if sim_e_exogenous[i,t] == 2.0:
                        sim_e[i,t] = 2.0
                        sim_ex[i,t] = 0.0
                        sim_ret_flag[i,t] = 0.0
                    else:
                        sim_e[i,t] = 0.0
                        sim_ex[i,t] = 0.0
                        retirement_age[i] = t
                        s_retirement[i] = sim_s[i,t]
                        sim_ret_flag[i,t] = 0.0

                # 1. technical variables
                if sim_ex[i,t] == 0.0:
                    if (sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 1.0) or (t == par.first_retirement and sim_e[i,t] == par.unemp): 
                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                        sim_h[i,t] = 0.0

                    elif sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 0.0: 
                        # 1.1 retirement age
                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                        sim_h[i,t] = 0.0

                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                        sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                    # 3.1 retirement payments 
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t
                    # 3.2 labor income 
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])

                else: 
                    # 1.1 retirement age
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])

                
                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)
                    
                    if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                        sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min
                    
                    # 3.1 retirement payments
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i],sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])


            elif t <= par.last_retirement:

                if sim_e[i,t-1] == 2.0: #Førtidspension eksisterere ikke længere og man skal overgå til pension
                    sim_e[i,t] = 2.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0

                elif sim_ex[i,t-1] == 1.0:
                    retirement_age[i] = t
                    s_retirement[i] = sim_s[i,t]
                    if sim_e_exogenous[i,t] == 2.0:
                        sim_e[i,t] = 2.0
                        sim_ex[i,t] = 0.0
                        sim_ret_flag[i,t] = 0.0
                    else:
                        sim_e[i,t] = 1.0
                        sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_ex[t,:,:,:,int(retirement_age[i]),int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                        sim_ex[i,t] = np.round(sim_ex[i,t])
                        sim_e[i,t] = sim_ex[i,t]
                        if sim_ex[i,t] == 0.0:
                            sim_ret_flag[i,t] = 1.0

                else: # just unemployed
                    sim_e[i,t] = 2.0
                    sim_ex[i,t] = 0.0
                    sim_ret_flag[i,t] = 0.0


                # 1. technical variables

                if sim_ex[i,t] == 0.0 or t == par.last_retirement:
                    if (sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 1.0) or (sim_ex[i,t-1] == 1.0 and t == par.last_retirement): 
                        # 1.1 retirement age
                        retirement_age[i] = t
                        s_retirement[i] = sim_s[i,t]

                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                        sim_h[i,t] = 0.0

                    elif sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 0.0: 
                        # 1.1 retirement age

                        # 2. Interpolation of choice variables
                        sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                        sim_h[i,t] = 0.0

                        # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                        sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                    # 3.1 retirement payments 
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t)
                    # 3.2 labor income 
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])


                else: 
                    # 1.1 retirement age
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_c[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid[t], sol_h[t,:,:,:,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])

                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)
                    
                    if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                        sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                    # 3.1 retirement payments
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t
                    # 3.2 labor income
                    sim_w[i,t] = np.minimum(wage(par, sim_k[i,t], t), par.w_max)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_e[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i],sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t], par.k_max[t])

            elif t > par.last_retirement:
                sim_ex[:,t] = 0.0
                sim_e[:,t]  = 2.0
                sim_ret_flag[i,t] = 0.0

                # 1.1 retirement age
                # 2. Interpolation of choice variables
                sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age[i]), int(sim_e[i,t])], sim_a[i,t], s_retirement[i])
                sim_h[i,t] = 0.0

                # 3. Income variables
                sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], sim_e[i,t], retirement_age[i], t)

                if sim_a[i,t] +sim_income[i,t] - sim_c[i,t] < par.a_min:    
                    sim_c[i,t] = sim_a[i,t] +sim_income[i,t] - par.a_min

                # 3.1 retirement payments
                sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, sim_h[i,t], s_retirement[i], sim_e[i,t], retirement_age[i], t) # par, h, s, e, r, t
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
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])

                elif par.T - 1 > t >= retirement_age[i] + par.m:
                    # 4. Update of states
                    # sim_a[i,t+1] = np.maximum(par.a_min, np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max))
                    sim_a[i,t+1] = np.minimum((1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t]), par.a_max)
                    sim_s[i,t+1] = np.minimum(np.maximum((sim_s[i,t] + sim_s_retirement_contrib[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i]))*(1+par.r_s), 0), par.s_max)
                    sim_k[i,t+1] = np.minimum(((1-par.delta)*sim_k[i,t])*sim_xi[i,t], par.k_max[t])
                    

    return sim_a, sim_s, sim_k, sim_c, sim_h, sim_w, sim_ex, sim_e, sim_chi_payment, sim_tax_rate, sim_income_before_tax_contrib, s_retirement, retirement_age, sim_income, sim_ret_flag 
