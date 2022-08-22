# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np
import scipy as sp

#sigma has to be entered in the same unit as the recoil energy (i.g. keV)
def gaussian(sigma,E,E_prime):
    #return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((E-E_prime)/sigma)**2)
    return np.exp(-0.5*((E-E_prime)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def convolve(sigma,E,E_prime,rate):
    """ Convolution of a certain recoil rate with a Gaussian energy resolution
        of sigma.
    
    Attributes
    ----------
    sigma : float
        The parameter sigma of the gaussian energy distribution.
    E : array
        Values of E for which we want the convoluted result.
    E_prime : array
        Array of energy values over which the convolution is integrated, has to be the same
        format as rate.
    rate : array
        Rate(E) which should be convoluted with the energy resolution, needs to be consistent (i.e. calculated from)
        with E_prime.
    """
    integral = np.zeros(np.shape(E))
    for i in range(0,np.shape(E)[0]):
        integrand = rate*gaussian(sigma,E[i],E_prime)
        integral[i] = np.trapz(integrand,E_prime)
    return integral
