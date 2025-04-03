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
    return par.full_time_hours*np.exp(np.log(par.w_0) + par.beta_1*k + par.beta_2*t**2)

# 1.1 The four sources of income all before taxes and retirement contributions - and total income before taxes and retirement contributions:
# 1.1.1 Capital income
@jit_if_enabled(fastmath=True)
def capital_return_fct(par, a):
    '''Capital return is the same for all periods'''
    return (par.r_a/(1+par.r_a)) * a

# 1.1.2. Retirement payouts
@jit_if_enabled(fastmath=True)
def calculate_retirement_payouts(par, savings, retirement_age, t):
    """Calculate retirement payouts: can be split into 3 periods: before retirement, during installment and annuity, and only annuity"""
    if t >= retirement_age + par.m:
        EL = sum(np.cumprod(par.pi_el[retirement_age:])*np.arange(retirement_age,par.T))/(par.T-retirement_age)
        s_retirement = savings
        s_lr = par.share_lr * (s_retirement/EL) 
        return s_lr, 0.0
    
    elif t >= retirement_age:
        EL = sum(np.cumprod(par.pi_el[retirement_age:])*np.arange(retirement_age,par.T))/(par.T-retirement_age)
        s_retirement = savings
        s_lr = par.share_lr * (s_retirement/EL) 
        s_rp = (1-par.share_lr) * (s_retirement/par.m) 
        return s_lr, s_rp
    else:
        return 0.0, 0.0

# 1.1.3. Labor income
@jit_if_enabled(fastmath=True)
def labor_income_fct(par, k, h, retirement_age, t):
    '''Before and after retirement age'''
    if t >= retirement_age:
        return 0.0
    else:
        return h*wage(par, k, t)

# 1.1.4. Public benefits
@jit_if_enabled(fastmath=True)
def public_benefit_fct(par, h, income, t):
    """Before retirement: unemployment benefits (if working, then no benefits), after retirement: public pension"""
    # Before public retirement age
    if t<par.retirement_age:
        if h==0.0:
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
def income_private_fct(par, a, s, k, h, retirement_age, t):
    '''Private income is taxed and is used for 
    Income before taxes and contribution: include capital return, retirement payouts, and wages'''
    # Capital income
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?

    # Retirement payouts
    if t >= retirement_age:
        s_lr, s_rp = calculate_retirement_payouts(par, s, retirement_age, t)
    else :
        s_lr, s_rp = 0.0, 0.0
    
    # labor income 
    labor_income = labor_income_fct(par, k, h, retirement_age, t)

    # Total income 
    total_income = labor_income + a_return + s_lr + s_rp

    # public benefits
    public_benefit = public_benefit_fct(par, h, total_income, t)

    return total_income + public_benefit

# 1.2 retirement contributions and tax rates:
# 1.2.1 tax rates: 
@jit_if_enabled(fastmath=True)
def tax_rate_fct(par, a, s, k, h, retirement_age, t):
    '''Tax rate as a function of income'''
    total_income = income_private_fct(par, a, s, k, h, retirement_age, t)
    
    if total_income <= par.threshold:
        tax_rate = par.L1*(1-np.exp(-par.K1*total_income))
    else:
        tax_rate = par.L1*(1-np.exp(-par.K1*total_income)) + par.L2*(1-np.exp(-par.K2*(total_income-par.threshold)))
    
    # Test 
    # tax_rate = par.upsilon 

    return tax_rate

# 1.2.2 retirement contributions, only of labor income 
@jit_if_enabled(fastmath=True)
def retirement_contribution_fct(par, a, s, k, h, retirement_age, t):
    '''Retirement contributions'''
    if t >= retirement_age:
        return 0.0
    else:
        return labor_income_fct(par, k, h, retirement_age, t)*par.tau[t]

# 1.3. calculate income after taxes and contributions
# 1.3.1 Income after taxes and contributions
@jit_if_enabled(fastmath=True)
def final_income_and_retirement_contri(par, a, s, k, h, retirement_age, t):
    ''' The following is taxed: capital income, retirement payouts, labor income, and public benefits
    retirement contributions are only of labor income'''
    # Incomes before taxes and contributions
    a_return = 0.0 # capital_return_fct(par, a), skal vi have capital return med?
    if t >= retirement_age:
        s_lr, s_rp = calculate_retirement_payouts(par, s, retirement_age, t)
    else:
        s_lr, s_rp = 0.0, 0.0
    
    labor_income = labor_income_fct(par, k, h, retirement_age, t)
    income_private = a_return + s_lr + s_rp + labor_income
    chi = public_benefit_fct(par, h, income_private, t)

    # Tax rate and retirement contribution
    tax_rate = tax_rate_fct(par, a, s, k, h, retirement_age, t)
    retirement_contribution = retirement_contribution_fct(par, a, s, k, h, retirement_age, t)
    if t >= retirement_age:
        # income after taxes&contributions
        return (1-tax_rate)*(income_private + chi), 0.0
    else:
        # income after taxes, and retirement contributions 
        return (1-tax_rate)*(income_private*(1-par.tau[t]) + chi), retirement_contribution


# 2. Helper functions in solving and optimizing
@jit_if_enabled(fastmath=True)
def budget_constraint(par, h, a, s, k, retirement_age, ex, t):
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, retirement_age, t)
    return par.c_min, max(par.c_min*2, a + income)

@jit_if_enabled(fastmath=True)
def precompute_EV_next(par, sol_V, retirement_idx, employed_idx, t):

    EV = np.zeros((len(par.a_grid), len(par.s_grid), len(par.k_grid)))

    V_next_un = sol_V[t+1, :, :, :, retirement_idx, 0]
    V_next_em = sol_V[t+1, :, :, :, retirement_idx, 1]

    if employed_idx ==0:
        V_next = (1-par.hire)*V_next_un + par.hire*V_next_em
    else:
        V_next = par.fire*V_next_un + (1-par.fire)*V_next_em

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
def calculate_last_period_consumption(par, a, s, retirement_age, t):
    k,h = 0.0,0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, retirement_age, t)
 
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
def value_last_period(par, c, a, s, retirement_age, t):
    # states and income 
    h, k = 0.0,0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, retirement_age, t)
    a_next = (1+par.r_a)*(a + income - c)

    return utility(par, c, h) + bequest(par, a_next)

@jit_if_enabled(fastmath=True)
def value_function_after_retirement(par, sol_V, c, a, s, retirement_age, e, t):
    # states and income 
    retirement_age_idx = retirement_age - par.first_retirement
    e_idx = int(e)
    h,k  = 0.0, 0.0
    income, _ = final_income_and_retirement_contri(par, a, s, k, h, retirement_age, t)

    # Next period states 
    a_next = (1+par.r_a)*(a + income - c)
    s_next = s
    V_next = sol_V[t+1, :, :, 0, retirement_age_idx, e_idx]
    EV_next = interp_2d(par.a_grid, par.s_grid, V_next, a_next, s_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


@jit_if_enabled(fastmath=True)
def value_function(par, sol_V, sol_EV, c, h, a, s, k, t):
    # states and income 
    income, retirement_contribution = final_income_and_retirement_contri(par, a, s, k, h, par.last_retirement, t)

    # Next period states
    a_next = (1+par.r_a)*(a + income - c)
    s_next = (1+par.r_s)*(s + retirement_contribution)
    k_next = ((1-par.delta)*k + h)
    EV_next = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_EV, a_next, s_next, k_next)

    return utility(par, c, h) + par.pi[t+1]*par.beta*EV_next + (1-par.pi[t+1])*bequest(par, a_next)


# 4. Objective functions 
@jit_if_enabled(fastmath=True)
def obj_consumption(c, par, sol_V, sol_EV, h, a, s, k, t):
    return -value_function(par, sol_V, sol_EV, c, h, a, s, k, t)

@jit_if_enabled()
def obj_consumption_after_retirement(c, par, sol_V, a, s, retirement_age, e, t):
    return -value_function_after_retirement(par, sol_V, c, a, s, retirement_age, e, t)

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

            for employed_idx in par.e_grid:
                employed = par.e_grid[employed_idx]

                if t <= retirement_age:
                    sol_EV = precompute_EV_next(par, sol_V, retirement_age_idx, employed_idx, t)

                for a_idx in prange(len(par.a_grid)):
                    assets = par.a_grid[a_idx]

                    for s_idx in range(len(par.s_grid)):
                        savings = par.s_grid[s_idx]

                        for k_idx in range(len(par.k_grid)):
                            human_capital = par.k_grid[k_idx]

                            idx = (t, a_idx, s_idx, k_idx, retirement_age_idx, employed_idx)
                            idx_next = (t+1, a_idx, s_idx, k_idx, retirement_age_idx, employed_idx)
                            
                            if t == par.T - 1: # Last period

                                sol_c[idx] = calculate_last_period_consumption(par, assets, savings, retirement_age, t)
                                sol_ex[idx] = ex_place
                                sol_h[idx] = hours_place
                                sol_V[idx] = value_last_period(par, sol_c[idx], assets, savings, retirement_age, t)

                            elif t > retirement_age: # After retirement age, with "ratepension"

                                bc_min, bc_max = budget_constraint(par, hours_place, assets, savings, human_capital_place, retirement_age, employed, t)

                                c_star = optimizer(
                                    obj_consumption_after_retirement,
                                    bc_min,
                                    bc_max,
                                    args=(par, sol_V, assets, savings, retirement_age, employed, t),
                                    tol=par.opt_tol
                                )

                                sol_c[idx] = c_star
                                sol_ex[idx] = ex_place
                                sol_h[idx] = hours_place
                                sol_V[idx] = value_function_after_retirement(par, sol_V, c_star, assets, savings, retirement_age, employed, t)
                    

                            # elif t == retirement_age and sol_ex[idx_next] == 0.0:
                            elif t == retirement_age and sol_ex[idx_next] == 0.0:  
                                for ex in (0, 1):
                                    if ex == 0.0: # Unemployed
                                        bc_min, bc_max = budget_constraint(par, hours_place, assets, savings, human_capital, retirement_age, ex, t)

                                        c_star_u = optimizer(
                                            obj_consumption_after_retirement,
                                            bc_min,
                                            bc_max,
                                            args=(par, sol_V, assets, savings, retirement_age, employed, t),
                                            tol=par.opt_tol
                                        )

                                        V_unemployed[idx] = value_function_after_retirement(par, sol_V, c_star_u, assets, savings, retirement_age, employed, t)
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

                                        if V_unemployed[idx] > V_employed[idx]:
                                            sol_V[idx]  = V_unemployed[idx]
                                            sol_ex[idx] = 0.0
                                            sol_h[idx]  = 0.0

                                        else:
                                            sol_V[idx]  = V_employed[idx]
                                            sol_ex[idx] = 1.0
                                            sol_h[idx]  = h_star


                            # elif t == retirement_age and sol_ex[idx_next] == 1.0:  
                            #     h_star = optimize_outer(
                            #         obj_hours,       
                            #         par.h_min,
                            #         par.h_max,
                            #         args=(par, sol_V, sol_EV, assets, savings, human_capital, retirement_age, ex, t),
                            #         tol=par.opt_tol
                            #     )

                            #     bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, retirement_age, ex, t)
                            #     c_star = optimizer(
                            #         obj_consumption,
                            #         bc_min,
                            #         bc_max,
                            #         args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                            #         tol=par.opt_tol
                            #     )

                            #     V_employed[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)

                            #     sol_c[idx]  = c_star
                            #     sol_h[idx]  = h_star
                            #     sol_V[idx]  = V_employed[idx]
                            #     sol_ex[idx] = 1.0

                            else:
                                # if employed == 0.0: # Forced unemployment
                                #     ex = 0.0
                                #     h_unemployed = 0.0
                                #     bc_min, bc_max = budget_constraint(par, h_unemployed, assets, savings, human_capital, par.last_retirement, ex, t)
                                #     c_star = optimizer(
                                #         obj_consumption,
                                #         bc_min,
                                #         bc_max,
                                #         args=(par, sol_V, sol_EV, h_unemployed, assets, savings, human_capital, t),
                                #         tol=par.opt_tol
                                #     )
                                #     V_unemployed[idx] = value_function(par, sol_V, sol_EV, c_star, h_unemployed, assets, savings, human_capital, t)
                                #     sol_c_un[idx]  = c_star
                                #     sol_h[idx]  = h_unemployed
                                    
                                # if employed == 1.0:
                                #     for ex in (0,1):
                                #         if ex == 0.0: # choose unemployment
                                #             h_unemployed = 0.0
                                #             bc_min, bc_max = budget_constraint(par, h_unemployed, assets, savings, human_capital, par.last_retirement, ex, t)
                                #             c_star = optimizer(
                                #                 obj_consumption,
                                #                 bc_min,
                                #                 bc_max,
                                #                 args=(par, sol_V, sol_EV, h_unemployed, assets, savings, human_capital, t),
                                #                 tol=par.opt_tol
                                #             )
                                #             V_unemployed[idx] = value_function(par, sol_V, sol_EV, c_star, h_unemployed, assets, savings, human_capital, t)
                                #             sol_c_un[idx]  = c_star
                                #             sol_h[idx]  = h_unemployed
                                            
                                #         if ex == 1.0: # choose employment
                                #             h_star = optimize_outer(
                                #                 obj_hours,       
                                #                 par.h_min,
                                #                 par.h_max,
                                #                 args=(par, sol_V, sol_EV, assets, savings, human_capital, par.last_retirement, ex, t),
                                #                 tol=par.opt_tol
                                #             )

                                #             bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, par.last_retirement, ex, t)
                                #             c_star = optimizer(
                                #                 obj_consumption,
                                #                 bc_min,
                                #                 bc_max,
                                #                 args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                #                 tol=par.opt_tol
                                #             )

                                            
                                #             V_employed[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)
                                #             sol_c[idx] = c_star
                                #             sol_h[idx] = h_star
                                #             if V_unemployed[idx]> V_employed[idx]:
                                #                 sol_V[idx] = V_unemployed[idx]
                                #                 sol_ex[idx] = 0.0
                                #             else:
                                #                 sol_V[idx] = V_employed[idx]
                                #                 sol_ex[idx] = 1.0

                                if employed == 0.0: # Forced unemployment
                                    ex = 0.0
                                    h_unemployed = 0.0
                                    bc_min, bc_max = budget_constraint(par, h_unemployed, assets, savings, human_capital, par.last_retirement, ex, t)
                                    c_star = optimizer(
                                        obj_consumption,
                                        bc_min,
                                        bc_max,
                                        args=(par, sol_V, sol_EV, h_unemployed, assets, savings, human_capital, t),
                                        tol=par.opt_tol
                                    )
                                    V_unemployed[idx] = value_function(par, sol_V, sol_EV, c_star, h_unemployed, assets, savings, human_capital, t)
                                    sol_c_un[idx]  = c_star
                                    sol_h[idx]  = 0.0

                                for ex in (0, 1):
                                    if ex == 0.0: # Unemployed
                                        h_unemployed = 0.0
                                        bc_min, bc_max = budget_constraint(par, h_unemployed, assets, savings, human_capital, par.last_retirement, ex, t)
                                        c_star = optimizer(
                                            obj_consumption,
                                            bc_min,
                                            bc_max,
                                            args=(par, sol_V, sol_EV, h_unemployed, assets, savings, human_capital, t),
                                            tol=par.opt_tol
                                        )
                                        V_unemployed[idx] = value_function(par, sol_V, sol_EV, c_star, h_unemployed, assets, savings, human_capital, t)
                                        sol_c_un[idx]  = c_star
                                        
                                    if ex == 1.0: # Employed
                                        h_star = optimize_outer(
                                            obj_hours,       
                                            par.h_min,
                                            par.h_max,
                                            args=(par, sol_V, sol_EV, assets, savings, human_capital, par.last_retirement, ex, t),
                                            tol=par.opt_tol
                                        )

                                        bc_min, bc_max = budget_constraint(par, h_star, assets, savings, human_capital, par.last_retirement, ex, t)
                                        c_star = optimizer(
                                            obj_consumption,
                                            bc_min,
                                            bc_max,
                                            args=(par, sol_V, sol_EV, h_star, assets, savings, human_capital, t),
                                            tol=par.opt_tol
                                        )

                                        V_employed[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)
                                        sol_c[idx] = c_star
                                        
                                        if V_unemployed[idx] > V_employed[idx]:
                                            sol_V[idx] = V_unemployed[idx]
                                            sol_ex[idx] = 0.0
                                            sol_h[idx] = 0.0
                                        
                                        else:
                                            sol_V[idx] = V_employed[idx]
                                            sol_ex[idx] = 1.0
                                            sol_h[idx] = h_star

                                        # if a_idx == 0 and s_idx == 0 and retirement_age_idx == 0 and t < 3:
                                        #     print(idx)
                                        #     print("V_uneV_unemployed[idx]mp", V_unemployed[idx])
                                        #     print("sol_c_un[idx]", sol_c_un[idx])
                                        #     print("V_employed[idx]", V_employed[idx])
                                        #     print("sol_c[idx]", sol_c[idx])


                                        
                                        # V_employed[idx] = value_function(par, sol_V, sol_EV, c_star, h_star, assets, savings, human_capital, t)
                                        # sol_c[idx] = c_star
                                        # sol_h[idx] = h_star

                                        # sol_V[idx] = V_unemployed[idx] if V_unemployed[idx]>V_employed[idx] else V_employed[idx]
                                        # sol_ex[idx] = 0.0 if V_unemployed[idx]>V_employed[idx] else 1.0

    return sol_c, sol_c_un, sol_h, sol_ex, sol_V

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
    sim_e_f = sim.e_f
    sim_e_h = sim.e_h
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
    sim_s_retirement_contrib = sim.s_retirement_contrib
    
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
        if t < par.first_retirement:
            for i in prange(par.simN):
                if t==0:
                    sim_e[i,0] = 1.0
                elif t < par.first_retirement:
                    if sim_e[i,t-1] == 1.0:
                        sim_e[i,t] = 0.0 if sim.e_f[i,t] == 1.0 else 1.0
                    else:
                        sim_e[i,t] = 1.0 if sim.e_h[i,t] == 1.0 else 0.0
                
                # 1. technical variables
                if sim_e[i,t] == 0.0:
                    sim_ex[i,t] = 0.0
                else:
                    sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_ex[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_ex[i,t] = np.maximum(0, np.round(sim_ex[i,t]))

                if sim_ex[i,t] == 0.0:
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c_un[t,:,:,:,int(retirement_age_idx[i]), int(sim_ex[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = 0.0

                    # 3. Income variables 
                    # 3.1 final income and retirement payments 
                    sim_income[i,t],sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], par.last_retirement, t)
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], par.last_retirement, t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = (1+par.r_s)*(sim_s[i,t] + sim_s_retirement_contrib[i,t])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t]

                if sim_ex[i,t] == 1.0:
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c[t,:,:,:,0, int(sim_ex[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_h[t,:,:,:,0, int(sim_ex[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])

                    # 3. Income variables 
                    # 3.1 final income and retirement payments 
                    sim_income[i,t],sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], par.last_retirement, t)
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], par.last_retirement, t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = (1+par.r_s)*(sim_s[i,t] + sim_s_retirement_contrib[i,t])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t]

        elif t <= par.last_retirement:
            for i in prange(par.simN):
                if t < par.last_retirement:
                    if sim_e[i,t-1] == 1.0:
                        sim_e[i,t] = 0.0 if sim.e_f[i,t] == 1.0 else 1.0
                    else:
                        sim_e[i,t] = 0.0
                # 1. technical variables
                if sim_e[i,t] == 0.0:
                    sim_ex[i,t] = 0.0
                else:
                    sim_ex[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_ex[t,:,:,:,int(retirement_age_idx[i]),int(sim_e[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_ex[i,t] = np.maximum(0, np.round(sim_ex[i,t]))

                if (sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 1.0) or (sim_ex[i,t-1] == 1.0 and t == par.last_retirement) or (t == par.first_retirement and sim_ex[i,t] == 0.0): 
                    # 1.1 retirement age
                    retirement_age[i] = t
                    retirement_age_idx[i] = t - par.first_retirement
                    s_retirement[i] = sim_s[i,t]

                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c_un[t,:,:,:,int(retirement_age_idx[i]),int(sim_ex[i,t])], sim_a[i,t], s_retirement[i], sim_k[i,t])
                    sim_h[i,t] = 0.0

                    # 3. Income variables
                    sim_income[i,t], _ = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    # 3.1 retirement payments 
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, s_retirement[i], retirement_age[i], t)
                    # 3.2 labor income 
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - (sim_s_lr_init[i]+sim_s_rp_init[i])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                elif sim_ex[i,t] == 0.0 and sim_ex[i,t-1] == 0.0: 
                    # 1.1 retirement age

                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i]), int(sim_ex[i,t])], sim_a[i,t], s_retirement[i])
                    sim_h[i,t] = 0.0

                    # 3. Income variables
                    # 3.1 retirement payments
                    sim_income[i,t], _ = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    sim_s_lr_init[i], sim_s_rp_init[i] = calculate_retirement_payouts(par, s_retirement[i], retirement_age[i], t)
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i],sim_k[i,t], sim_h[i,t], retirement_age[i], t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                else: 
                    # 1.1 retirement age
                    # 2. Interpolation of choice variables
                    sim_c[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_c[t,:,:,:,t-par.first_retirement, int(sim_ex[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])
                    sim_h[i,t] = interp_3d(par.a_grid, par.s_grid, par.k_grid, sol_h[t,:,:,:,t-par.first_retirement, int(sim_ex[i,t])], sim_a[i,t], sim_s[i,t], sim_k[i,t])

                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], par.last_retirement, t)
                    # 3.1 retirement payments
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], par.last_retirement, t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], par.last_retirement, t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = (1+par.r_s)*(sim_s[i,t] + sim_s_retirement_contrib[i,t])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t] + sim_h[i,t])*sim_xi[i,t]          

        elif t > par.last_retirement:
            sim_ex[:,t] = 0.0

            for i in prange(par.simN):
                # 1.1 retirement age

                # 2. Interpolation of choice variables
                sim_c[i,t] = interp_2d(par.a_grid, par.s_grid, sol_c[t,:,:,0,int(retirement_age_idx[i]), int(sim_ex[i,t])], sim_a[i,t], s_retirement[i])
                sim_h[i,t] = 0.0

                if t < retirement_age[i] + par.m: 
                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    # 3.1 retirement payments
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i],sim_k[i,t], sim_h[i,t], retirement_age[i], t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - (sim_s_lr_init[i] + sim_s_rp_init[i])
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]

                elif par.T - 1 > t >= retirement_age[i] + par.m:
                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    # 3.1 retirement payments
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], s_retirement[i],sim_k[i,t], sim_h[i,t], retirement_age[i], t)

                    # 4. Update of states
                    sim_a[i,t+1] = (1+par.r_a)*(sim_a[i,t] + sim_income[i,t] - sim_c[i,t])
                    sim_s[i,t+1] = sim_s[i,t] - sim_s_lr_init[i]
                    sim_k[i,t+1] = ((1-par.delta)*sim_k[i,t])*sim_xi[i,t]
                    
                else:
                    # 3. Income variables
                    sim_income[i,t], sim_s_retirement_contrib[i,t] = final_income_and_retirement_contri(par, sim_a[i,t], s_retirement[i], sim_k[i,t], sim_h[i,t], retirement_age[i], t)
                    # 3.1 retirement payments
                    # 3.2 labor income
                    sim_w[i,t] = wage(par, sim_k[i,t], t)
                    # 3.3 public benefits
                    sim_chi_payment[i,t] = public_benefit_fct(par, sim_h[i,t], sim_income[i,t], t)
                    # 3.4 income before tax contribution
                    sim_income_before_tax_contrib[i,t] = income_private_fct(par, sim_a[i,t], sim_s[i,t], sim_k[i,t], sim_h[i,t], retirement_age[i], t) 
                    # 3.5 tax rate
                    sim_tax_rate[i,t] = tax_rate_fct(par, sim_a[i,t], sim_s[i,t],sim_k[i,t], sim_h[i,t], retirement_age[i], t)


    return sim_a, sim_s, sim_k, sim_c, sim_h, sim_w, sim_ex, sim_chi_payment, sim_tax_rate, sim_income_before_tax_contrib, s_retirement, retirement_age
