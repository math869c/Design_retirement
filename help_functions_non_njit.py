import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, lognorm, norm

def draw_initial_values(simN):
    # load data
    means_data = pd.read_csv("Data/mean_matrix.csv")
    covariance_matrix = pd.read_csv("Data/variance_matrix_30.csv")
    means = np.array(means_data[["formue_plsats_Mean", "pension_u_skat_plsats_Mean", "hourly_salary_plsats_Mean"]])[0]
    covariance_matrix.set_index('_NAME_', inplace=True)
    covar_array = np.array(covariance_matrix.loc[["formue_plsats", "pension_u_skat_plsats", "hourly_salary_plsats"], ["formue_plsats", "pension_u_skat_plsats", "hourly_salary_plsats"]].values)

    # calculate normal parameters
    outer_means = np.outer(means, means)
    covar_normal = np.log(covar_array / outer_means + 1)
    means_normal = np.log(means)-0.5*np.diag(covar_normal)

    # draw from normal distribution and convert back to lognormal
    dist = multivariate_normal(means_normal, covar_normal, seed=1)
    draws = np.exp(dist.rvs(size=simN))

    return draws[:,0], draws[:,1], draws[:,2]

def logistic(x, L, f, x0):
    return L / (1 + np.exp(-f * (x - x0)))




def compute_transition_probs(x, param_table):
    eta = {1: 0.0, 2: 0.0}
    for _, row in param_table.iterrows():
        var = row["Variable"]
        response = row["Response"]
        estimate = row["Estimate"]

        if var == "Intercept":
            eta[response] += estimate
        else:
            if var in x:
                eta[response] += estimate * x[var]
            elif "*" in var:
                terms = var.split("*")
                val = np.prod([x.get(t, 0) for t in terms])
                eta[response] += estimate * val

    exp_eta1 = np.exp(eta[1])
    exp_eta2 = np.exp(eta[2])
    denom = 1 + exp_eta1 + exp_eta2

    return {
        0: 1 / denom,
        1: exp_eta1 / denom,
        2: exp_eta2 / denom
    }

def eksog_prob(par, parameter_table_with_control):
    # Age range
    age_end = par.last_retirement + 1
    ages = np.arange(par.T)

    # DataFrame to collect results
    results = []

    # Compute for e_state_lag = 0 and 1
    for e_state_lag in [0, 1]:
        for age in ages:
            x_input = {
                "e_state_lag": e_state_lag,
                "alder": age,
                "alder2": age**2,
                "dummy_60_65": int(age_end-11 <= age ),
                "alder*e_state_lag": age * e_state_lag,
                "alder2*e_state_lag": (age**2) * e_state_lag,
                "dummy_60_*e_state_la": int(age_end-11 <= age) * e_state_lag
            }
            # 👉 Force P(e=2) = 0 at age 35
            probs = compute_transition_probs(x_input, parameter_table_with_control)

            if e_state_lag == 0:
                if age >= par.retirement_age:
                    # Must go on early retirement
                    probs[0] = 0.0
                    probs[1] = 0.0
                    probs[2] = 1.0
                elif age >= par.first_retirement:
                    # Cannot be hired
                    probs[1] = 0.0
                    total = probs[0] + probs[2]
                    probs[0] /= total
                    probs[2] /= total

            elif e_state_lag == 1:
                if age>= par.last_retirement:
                    probs[0] = 0.0
                    probs[1] = 0.0
                    probs[2] = 1.0
                elif age >= par.retirement_age:
                    # Move all P(e=1) mass to P(e=2)
                    probs[2] = probs[0]
                    probs[0] = 0.0
                    total = probs[1] + probs[2]
                    probs[1] /= total
                    probs[2] /= total

            results.append({
                "age": age,
                "e_state_lag": e_state_lag,
                "P_0": probs[0],
                "P_1": probs[1],
                "P_2": probs[2]
            })


    # Convert to DataFrame
    return pd.DataFrame(results)