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
