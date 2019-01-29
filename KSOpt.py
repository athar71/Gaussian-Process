#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:43:17 2019

@author: athar
"""

import numpy as np
import gp as gps
#import plotters as plts
import sklearn.gaussian_process as gp

def ycompute(X, norm = 2):
    exp_values = np.array([5.83e-5, 6.39e-4, 6.13e-4])
    return -(np.linalg.norm((X - exp_values), ord=norm))**2


    
bounds = np.array([[80, 120],[0.0003, 0.0007]])

xp = np.array([[80,0.0007],[100,5.00e-4],[120,3.00e-4]])
X_exp = np.array([[6.43e-5,4.22e-4, 5.04e-4],[4.97e-5,4.15e-4, 4.79e-4], [3.73e-5,4.19e-4, 4.43e-4]])
yp = np.array([ycompute(i) for i in X_exp])   

n_params = bounds.shape[0]

# Define the GP
kernel = gp.kernels.Matern()
#kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
#kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
model = gp.GaussianProcessRegressor(kernel=kernel,
                                      alpha=1e-4,
                                      n_restarts_optimizer=10,
                                      normalize_y=True)


model.fit(xp, yp)
next_sample = gps.sample_next_hyperparameter(gps.expected_improvement, model, yp, greater_is_better=True, bounds=bounds, n_restarts=500)

# Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
if np.any(np.abs(next_sample - xp) <= 1e-7):
    next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
