
# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

# This sheet contains all necessary data, such as masses and velocities, the
# classes are mainly for appropriate namespaces but also to define alternative 
# data sets. For example use
#   $ altcosmo = cosmo
#   $ altcosmo.v_esc = 650 
# to use a different set of cosmological constant.

class cosmo:
# necesarry velocities in km/s:
    c = 2.99792458e5
    v_esc = 544   #galactic escape velocity
    w = 270  #root mean square velocity in the glactic halo
    v_earth = 220*(1.05+0.07)  #velocity of earth with respect to the galactiv rest frame

class atomic:
    #elementary charge in C
    qe = 1.602176462e-19
    hred = 6.582119569e-16

    #mass numbers dictionary:
    A = {
    "Na" : 23.0,
    "I" : 127.0,
    "W" : 184.0,
    "Ca" : 40.0,
    "O" : 16.0,
    "Al" : 27.0,
    "Xe" : 131.0,
    "Ge" : 72.6,
    }

    A_Na = 23.0
    A_I = 127.0
    A_W = 184.0
    A_Ca = 40.0
    A_O = 16.0
    A_Al = 27.0
    A_Xe = 131.0
    A_Ge = 72.6
    A_Si = 28.085

    #atomic mass unit in GeV/c^2
    amu = 0.931494028 

    #masses in GeV/c^2
    m_p = 0.9382720

class darkmatter:
    #in pb:
    sigma_nuc = 1

    #in GeV/c^2/cm^3:
    rho_wimp= 0.3

    #mass in GeV/c^2
    m_wimp = 10 


class conversion:
    #unit-conversion factor for differential event rate in 1/kg keV d pb
    conv = 1e-52*(cosmo.c*1e3)**4*atomic.qe**-1*86400

    #unit-conversion factor for differential event rate in 1/kg keV d for a model
    #where the coupling is a direct input paramter and dimensionless
    conv_g = 86400*(cosmo.c*1e3)**6*atomic.hred**2*atomic.qe**-1*1e-18 
