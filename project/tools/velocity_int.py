# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np
import scipy.special as sp
from project.constants import cosmo 


z = np.sqrt((3*cosmo.v_esc**2)/(2*cosmo.w**2))
N = sp.erf(z)-2/(np.sqrt(np.pi))*z*np.exp(-z**2)
eta = np.sqrt((3*cosmo.v_earth**2)/(2*cosmo.w**2))
fac = 1/(N*eta)*(3/(2*np.pi*cosmo.w**2))**0.5


def mu_N(m_N,m_dm):
    return (m_dm*m_N)/(m_dm+m_N)

def x_min(vmin):
    return np.sqrt((3*vmin**2)/(2*cosmo.w**2))

def v_min(m_N,m_dm,E):
    return np.sqrt((E*1e-6*m_N)/(2*mu_N(m_N,m_dm)**2))*cosmo.c

def I(vmin):
    integral = np.zeros(np.size(vmin))
    for j in range(0,np.size(vmin)):
        if x_min(vmin[j])<(z-eta):
            integral[j] = fac*(0.5*np.sqrt(np.pi)*(sp.erf(x_min(vmin[j])+eta)-sp.erf(x_min(vmin[j])-eta))-2*eta*np.exp(-z*z))
        elif x_min(vmin[j])<(z+eta):
            integral[j] = fac*(0.5*np.sqrt(np.pi)*(sp.erf(z)-sp.erf(x_min(vmin[j])-eta))-np.exp(-z**2)*(z+eta-x_min(vmin[j])))
        else:
            integral[j] = 0
    return integral

