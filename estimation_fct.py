import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
 

def prepare_data_old(par):
    # load data
    means_data = pd.read_csv("Data/mean_matrix.csv")
    covariance_matrix = pd.read_csv("Data/variance_matrix.csv")

    # Concate to the final moments
    assets = np.array(means_data["formue_plsats_Mean"])
    savings = np.array(means_data["pension_u_skat_plsats_Mean"])
    hours = np.array(means_data["yearly_hours_Mean"])/par.full_time_hours
    extensive = np.array(means_data["extensive_v2_Mean"])
    hours = hours[:40]
    extensive = extensive[:40]

    mean = np.concatenate([extensive,
                        assets, 
                        savings, 
                        hours])
    moments = {'extensive':extensive, 'assets':assets, 'savings':savings, 'hours':hours}

    # Alter covariance matrix to create the final weighting matrix
    numeric_cols = covariance_matrix.select_dtypes(include=[np.number]).columns[1:]

    row_mask = covariance_matrix["_NAME_"].str.startswith("yearly_hours")
    col_mask = [col for col in covariance_matrix.columns if col.startswith("yearly_hours")]

    covariance_matrix.loc[row_mask, numeric_cols] = covariance_matrix.loc[row_mask, numeric_cols].astype(float)

    covariance_matrix.loc[row_mask, numeric_cols] /= par.full_time_hours

    covariance_matrix[col_mask] /= par.full_time_hours

    # Filter extensive margin to only include first 40 ages
    ext_rows = covariance_matrix["_NAME_"].str.startswith("extensive_v2")
    ext_indices = covariance_matrix[ext_rows].index[:40]  # First 40 rows for extensive

    # Keep only the rows and columns corresponding to your full moment vector (175 elements)
    # Assumes order: assets (70), savings (70), hours (40), extensive (40) → total = 220
    # So we filter the covariance matrix to be 220×220 in same order
    total_keep = 70 + 70 + 40 + 40  # = 220

    # Now get only the top-left 220×220 block
    cov_matrix_numeric = covariance_matrix[numeric_cols].to_numpy()
    cov_matrix_numeric = cov_matrix_numeric[:total_keep, :total_keep]

    # Identify good (non-NaN) moments
    valid = ~np.isnan(np.diag(cov_matrix_numeric))

    # Keep only valid rows and columns
    cov_matrix_numeric = cov_matrix_numeric[valid][:, valid]
    mean = mean[valid]

    # Final weighting matrix
    weights = np.linalg.pinv(cov_matrix_numeric)

    return mean, weights, moments

def prepare_data(par, year="all"):
    # Load data

    if year == "all":
        means_data = pd.read_csv("Data ny def/mean_matrix.csv")
        covariance_matrix = pd.read_csv("Data ny def/variance_matrix.csv")

    elif year == 2019:
        means_data = pd.read_csv("Data 2019/mean_matrix.csv")
        covariance_matrix = pd.read_csv("Data 2019/variance_matrix.csv")

    else:
        assert False

    # Process means
    assets = np.array(means_data["formue_plsats_Mean"])
    savings = np.array(means_data["pension_u_skat_plsats_Mean"])
    hours = np.array(means_data["yearly_hours_Mean"]) / par.full_time_hours
    extensive = np.array(means_data["extensive_v2_Mean"])
    hours = hours[:45]
    extensive = extensive[:45]

    mean = np.concatenate([extensive, assets, savings, hours])

    # Drop invalid moments
    covariance_matrix = pd.read_csv("Data ny def/variance_matrix.csv")
    variance_diag = np.diag(covariance_matrix.iloc[:,2:])
    variance_diag = variance_diag[~np.isnan(variance_diag)] 

    # Construct diagonal weighting matrix: 1/variance
    safe_variances = np.where(variance_diag == 0, 1e-6, variance_diag)  # Avoid divide-by-zero
    safe_variances[-55:] = safe_variances[-55:] / (par.full_time_hours**2)

    # safe_variances[:30] = safe_variances[:30] / 2

    weights = np.diag(1.0 / safe_variances)

    return mean, weights, {
        'extensive': extensive, 'assets': assets, 'savings': savings, 'hours': hours
    }


def load_and_process_data(mean_file, var_file, par, variables):
    # Load data
    means_data = pd.read_csv(mean_file)
    covariance_matrix = pd.read_csv(var_file)

    # Extract and process means
    mean_vectors = []
    for var in variables:
        mean_col = f"{var}_Mean"
        vec = np.array(means_data[mean_col])
        
        if var == "yearly_hours":
            vec = vec / par.full_time_hours
            vec = vec[:45]
        elif var == "extensive_v2":
            vec = vec[:45]

        mean_vectors.append(vec)

    mean = np.concatenate(mean_vectors)

    # Build row/column masks
    row_mask = covariance_matrix["_NAME_"].str.startswith(tuple(variables))
    col_mask = [col for col in covariance_matrix.columns if col.startswith(tuple(variables))]

    # Adjust for scale if hours included
    if "yearly_hours" in variables:
        hours_row_mask = covariance_matrix["_NAME_"].str.startswith("yearly_hours")
        hours_col_mask = [col for col in covariance_matrix.columns if col.startswith("yearly_hours")]

        covariance_matrix.loc[hours_row_mask, hours_col_mask] /= par.full_time_hours**2

    # Subset covariance matrix
    subset_matrix = covariance_matrix.loc[row_mask, col_mask]

    # Extract diagonal
    variance_diag = np.diag(subset_matrix)
    variance_diag = variance_diag[~np.isnan(variance_diag)]

    return mean, variance_diag



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
    # avg_s_by_age = np.mean(sim_data.s, axis=0)[:55]  # Length 70
    avg_h_by_age = np.nan_to_num(np.nanmean(np.where(sim_data.ex == 1, sim_data.h, np.nan), axis=0)[:45], nan=0.0) # Length 40
    avg_ex_by_age = np.mean(sim_data.ex, axis=0)[:45]  # Length 40

    # Concatenate and return
    return np.concatenate((avg_h_by_age, avg_a_by_age, avg_ex_by_age))


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
            print_str += f'{theta[i]:2.10f}, '
        print(print_str)
    
    # Compute simulated moments
    mom_sim = simulate_moments(theta, theta_names, model)

    # Sum of squared errors over all 175 elements
    obj = (mom_data - mom_sim).T @ W @ (mom_data - mom_sim)

    end_time = time.time()  # End timing
    elapsed_time = end_time - start_time
    
    if do_print: 
        print(f"Error = {obj:.5f}, Time = {elapsed_time:.1f} seconds")

    return obj

