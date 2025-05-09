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

def eksog_prob_simpel(par):
    df = pd.read_csv("Data/parameter_estiamtes_simpel.csv")[['Variable', 'Response',  'Estimate']]
    everything = []
    total_1 = []
    total_2 = []

    for x in range(par.T):
        eta = {1: 0.0, 2: 0.0}

        for e_state in [1.0, 2.0]: 
            df_state = df[df['Response'] == e_state]
            for _, row in df_state.iterrows():
                var = row["Variable"]
                estimate = row["Estimate"]
                
                if var == 'Intercept':
                    eta[e_state] = estimate
                elif var == 'alder':
                    eta[e_state] += estimate * x
                # elif var == 'alder2':
                #     eta[e_state] += estimate * x**2
                elif var == "dummy_60_65":
                    if x >= par.first_retirement:
                        eta[e_state] += estimate

        total_1.append(eta[1])
        total_2.append(eta[2])


    group_0 = []
    group_1 = []    
    group_2 = []

    for x in range(par.T):
        # if 40 > x >= 35:
        #     group_0.append(0.0)
        #     group_1.append(np.exp(total_1[x])/(np.exp(total_1[x]) + np.exp(total_2[x])))
        #     group_2.append(np.exp(total_2[x])/(np.exp(total_1[x]) + np.exp(total_2[x])))
        if x >= par.last_retirement:
            group_0.append(0.0)
            group_1.append(0.0)
            group_2.append(1.0)
        else:
            group_0.append(1/(1+np.exp(total_1[x]) + np.exp(total_2[x])))
            group_1.append(np.exp(total_1[x])/(1+np.exp(total_1[x]) + np.exp(total_2[x])))
            group_2.append(np.exp(total_2[x])/(1+np.exp(total_1[x]) + np.exp(total_2[x])))

    probabilites =  {'to_0': group_0, 'to_1': group_1, 'to_2': group_2}
    everything.append(probabilites)
    return everything

