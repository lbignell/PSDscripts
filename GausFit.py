# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 14:39:20 2015

@author: lbignell
"""
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def DoubleGauss(x, *p):
    A1, mu1, sigma1, A2, mu2, sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))

bin_centres = (thehist[2][:-1] + thehist[2][1:])/2
p0 = [100, 0.62, 0.03, 100, 0.7, 0.03]
coeff, var_matrix = curve_fit(DoubleGauss, bin_centres, sum(np.transpose(thehist[0][:]), 1), p0)
fitted = DoubleGauss(bin_centres, *coeff)
plt.plot(bin_centres,sum(np.transpose(thehist[0][:]),1),label='data')
plt.plot(bin_centres,fitted,label='fit')