# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np

#skin thickness in fm
s=1

#nuclear radius in fm
def R(A):
    return 1.2*A**(1/3)

def j1(x):
    return (np.sin(x)-x*np.cos(x))/x**2

def F2(A,m,E):
    R0 = np.sqrt(R(A)**2-5*s**2)
    q = np.sqrt(2*m*E*1e-6)/0.1975 #momentum transfer in 1/fm, 1/GeV=0.1975 fm
    return (3*j1(q*R0)/(q*R0))**2*np.exp(-0.5*q**2*s**2)




