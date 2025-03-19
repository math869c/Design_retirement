import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
 

def prepare_data(par):
    # load data
    means_data = pd.read_csv("Data/mean_matrix.csv")
    covariance_matrix = pd.read_csv("Data/variance_matrix.csv")

    # Concate to the final moments
    assets = np.array(means_data["formue_2018_Mean"])
    savings = np.array(means_data["pension_2018_Mean"])
    hours = np.array(means_data["yearly_hours_Mean"])/par.full_time_hours
    mean = np.concatenate([assets, 
                        savings, 
                        hours])
    moments = {'assets':assets, 'savings':savings, 'hours':hours}

    # Alter covariance matrix to create the final weighting matrix
    numeric_cols = covariance_matrix.select_dtypes(include=[np.number]).columns[1:]

    row_mask = covariance_matrix["_NAME_"].str.startswith("yearly_hours")
    col_mask = [col for col in covariance_matrix.columns if col.startswith("yearly_hours")]

    covariance_matrix.loc[row_mask, numeric_cols] = covariance_matrix.loc[row_mask, numeric_cols].astype(float)

    covariance_matrix.loc[row_mask, numeric_cols] /= par.full_time_hours

    covariance_matrix[col_mask] /= par.full_time_hours

    weights = np.linalg.inv(np.nan_to_num(covariance_matrix[numeric_cols].to_numpy(), nan=0))

    return mean, weights, moments




def scale_params(theta, bounds):
    """
    Scale theta to [0,1] given the bounds.
    """
    scaled = []
    for val, (low, high) in zip(theta, bounds):
        scaled_val = (val - low) / (high - low)
        scaled.append(scaled_val)
    return np.array(scaled)

def unscale_params(scaled_theta, bounds):
    """
    Convert scaled_theta in [0,1] back to original bounds.
    """
    unscaled = []
    for val, (low, high) in zip(scaled_theta, bounds):
        unscaled_val = low + val * (high - low)
        unscaled.append(unscaled_val)
    return np.array(unscaled)

def moment_func(sim_data):
    # Compute age-averaged moments
    avg_a_by_age = np.mean(sim_data.a, axis=0)  # Length 70
    avg_s_by_age = np.clip(np.mean(sim_data.s, axis=0), 0, None)  # Length 70
    avg_h_by_age = np.mean(sim_data.h, axis=0)  # Length 70

    # Concatenate and return
    return np.concatenate((avg_a_by_age, avg_s_by_age, avg_h_by_age))


def simulate_moments(theta, theta_names, model):
        
    # 1. Update model parameters
    for i, name in enumerate(theta_names):
        setattr(model.par, name, theta[i])
    
    # 2. Solve and simulate the model
    model.solve()
    model.simulate()
    
    # 3. Return the expanded vector of simulated moments
    return moment_func(model.sim)

def obj_func(scaled_theta, theta_names, mom_data, W, model, bounds, do_print=False):
    start_time = time.time()  # Start timing

    theta = unscale_params(scaled_theta, bounds)

    if do_print: 
        print_str = ''
        for i, name in enumerate(theta_names):
            print_str += f'{name}={theta[i]:2.3f} '
        print(print_str)
    
    # Compute simulated moments
    mom_sim = simulate_moments(theta, theta_names, model)

    # Sum of squared errors over all 175 elements
    obj = (mom_data - mom_sim).T @ W @ (mom_data - mom_sim)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    
    if do_print: 
        print(f"Error = {obj:.6f}, Time = {elapsed_time:.4f} seconds")

    return obj

