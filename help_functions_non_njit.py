import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, lognorm, norm

def draw_initial_values(Nsize):
    # load data
    means_data = pd.read_csv("Data/mean_matrix.csv")
    covariance_matrix = pd.read_csv("Data/variance_matrix.csv")
    means = np.array(means_data[["formue_2018_Mean", "formue_2018_Mean", "formue_2018_Mean"]])[0]
    covar_array = np.array(covariance_matrix[pd.notna(covariance_matrix['formue_2018_30'])].dropna(axis=1, how='all').iloc[0:3,2:5])
    
    # calculate normal parameters
    outer_means = np.outer(means, means)
    covar_normal = np.log(covar_array / outer_means + 1)
    means_normal = np.log(means)-0.5*np.diag(covar_normal)

    # draw from normal distribution and convert back to lognormal
    dist = multivariate_normal(means_normal[1:], covar_normal[1:,1:], seed=1)
    draws = np.exp(dist.rvs(size=Nsize))
    dist_formue = norm(means[0], covar_normal[0,0])
    draws_formue = np.abs(dist_formue.rvs(size=Nsize))
    return draws_formue, draws[:,0], draws[:,1]