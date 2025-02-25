# -*- coding: utf-8 -*-
"""golden section search

Numba JIT compilled golden section search optimizer for a custom objective.
"""

import math
import numpy as np
from jit_module import jit_if_enabled
from consav.linear_interp import interp_1d, interp_2d, interp_3d
from numba import prange

@jit_if_enabled()
def optimizer(obj,a,b,args=(),tol=1e-6):
    """ golden section search optimizer
    
    Args:

        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double,optional): tolerance

    Returns:

        (float): optimization result
    
    """
    
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2     
        
    # a. distance
    dist = b - a
    if dist <= tol: 
        return (a+b)/2

    # b. number of iterations
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c,*args)
    yd = obj(d,*args)

    # d. loop
    for _ in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi*dist
            c = a + inv_phi_sq * dist
            yc = obj(c,*args)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi*dist
            d = a + inv_phi * dist
            yd = obj(d,*args)

    # e. return
    if yc < yd:
        return (a+d)/2
    else:
        return (c+b)/2
    
@jit_if_enabled()
def optimize_outer(obj,a,b,args=(),tol=1e-6):
    """ golden section search optimizer
    
    Args:

        obj (callable): 1d function to optimize over
        a (double): minimum of starting bracket
        b (double): maximum of starting bracket
        args (tuple): additional arguments to the objective function
        tol (double,optional): tolerance

    Returns:

        (float): optimization result
    
    """
    
    inv_phi = (np.sqrt(5) - 1) / 2 # 1/phi                                                                                                                
    inv_phi_sq = (3 - np.sqrt(5)) / 2 # 1/phi^2     
        
    # a. distance
    dist = b - a
    if dist <= tol: 
        return (a+b)/2

    # b. number of iterations
    n = int(np.ceil(np.log(tol/dist)/np.log(inv_phi)))

    # c. potential new mid-points
    c = a + inv_phi_sq * dist
    d = a + inv_phi * dist
    yc = obj(c,*args,dist)
    yd = obj(d,*args,dist)

    # d. loop
    for _ in range(n-1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            dist = inv_phi*dist
            c = a + inv_phi_sq * dist
            yc = obj(c,*args,dist)
        else:
            a = c
            c = d
            yc = yd
            dist = inv_phi*dist
            d = a + inv_phi * dist
            yd = obj(d,*args,dist)

    # e. return
    if yc < yd:
        return (a+d)/2
    else:
        return (c+b)/2
    

@jit_if_enabled(parallel=True, fastmath=True)
def interp_3d_vec(grid1,grid2,grid3,value,xi1,xi2,xi3,yi):
    """ 3d interpolation for vector of points
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (3d)
        xi1 (numpy.ndarray): input vector
        xi2 (numpy.ndarray): input vector
        xi3 (numpy.ndarray): input vector
        yi (numpy.ndarray): output vector

    """

    for i in prange(yi.size):
        yi[i] = interp_3d(grid1,grid2,grid3,value,xi1[i],xi2[i],xi3[i])