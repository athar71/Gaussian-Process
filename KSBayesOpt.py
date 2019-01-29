#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 19:55:49 2019

@author: athar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern,RBF
from bayesian_optimization_util import plot_approximation, plot_acquisition


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    ''' Computes the EI at points X based on existing samples X_sample and Y_sample using a Gaussian process surrogate model. Args: X: Points at which EI shall be computed (m x d). X_sample: Sample locations (n x d). Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. xi: Exploitation-exploration trade-off parameter. Returns: Expected improvements at points X. '''
    mu, sigma = gpr.predict(X, return_std=True)
    #mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, X_sample.shape[1])
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [...]
    #mu_sample_opt = np.max(mu_sample)
    mu_sample_opt = np.max(Y_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    ''' Proposes the next sampling point by 
    optimizing the acquisition function. 
    Args: acquisition: Acquisition function. 
    X_sample: Sample locations (n x d).
    Y_sample: Sample values (n x 1). 
    gpr: A GaussianProcessRegressor fitted to samples. 
    Returns: Location of the acquisition function maximum. '''
    
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


def ycompute(X, norm = 2):
    exp_values = np.array([5.83e-5, 6.39e-4, 6.13e-4])
    return -np.linalg.norm((X - exp_values), ord=norm)
    
bounds = np.array([[80, 120],[0.0003, 0.0007]])
noise = 1e-3
X_sample = np.array([[60,0.0007]])
X_exp = np.array([[1.36e-4,6.77e-4, 7.70e-4]])
Y_sample = np.array([ycompute(X_exp)]) 

# Gaussian process with Mat??rn kernel as surrogate model
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
#m52 = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

# Update Gaussian process with existing samples
gpr.fit(X_sample, Y_sample)

# Obtain next sampling point from the acquisition function (expected_improvement)
X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds) 