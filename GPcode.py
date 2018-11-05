#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 10:28:02 2018

@author: athar
"""

import numpy as np
import matplotlib.pylab as plt

np.random.seed(42)
fixed_Xs =np.arange(-3,3,0.5)

#This function computes the RBF kernel Function k(x,x_prime) 
#&= theta_1* exp\left({\theta_2}*0.5(x-x^{\prime})^2\right)
#params[0] is theta1 which is variance - params[1] is 1/l

def exponential_cov(x, y, params):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)
    #return params[0] * np.exp( -0.5 * params[1] * numpy.linalg.norm( np.subtract(x, y))**2)
    
def conditional(x_new, x, y, params):
    C = exponential_cov(x_new, x, params)
    A = exponential_cov(x, x, params)
    B = exponential_cov(x_new, x_new, params)
    mu = np.linalg.inv(A).dot(C.T).T.dot(y)
    sigma = B - C.dot(np.linalg.inv(A).dot(C.T))
    return(mu.squeeze(), sigma.squeeze())

 # x is new point, data is the points that we already have   
def predict(x, data, kernel, params, sigma, t):
    k = [kernel(x, y, params) for y in data] # k is K_star
    Sinv = np.linalg.inv(sigma) # Sinv is K_inverse
    y_pred = np.dot(k, Sinv).dot(t)# t is y for the observed points
    sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
    return y_pred, sigma_new    

θ = [1, 1]
 #select an arbitrary  
x = [fixed_Xs[0]]
σ_0 = exponential_cov(x, x, θ)
sigma = [σ_0]
y=[float(np.random.normal(scale=σ_0))]
sigma_neg=[]
σ_inv = []
K_star = []
σ = []
for x_pred in fixed_Xs[1:]:
    
    
    σ_1 = exponential_cov(x, x, θ)
    σ_inv.append(np.linalg.inv([σ_1]))
    σ.append([σ_1])
    
    K_star.append([exponential_cov(x_pred, x, θ)])
    
    
    #predictions = conditional(x_pred, x, y, θ)
    predictions = predict(x_pred, x, exponential_cov,θ,σ_1,y)
    y_pred, sigmas = np.transpose(predictions)
    x.append(x_pred)
    if (sigmas>=0 ):
        y.append(float(np.random.normal(y_pred, sigmas)))
        sigma.append(sigmas)
        
    else: 
        y.append(y_pred)
        sigma.append(0)
        sigma_neg.append(sigmas)


#np.dot(K_star[50],σ_inv[50]).dot (np.transpose(K_star[50]))    
plt.errorbar(x, y, yerr=sigma, capsize=0)
plt.plot(x,y,'ro')
plt.show()