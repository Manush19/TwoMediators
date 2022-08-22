"""
@author: einfaltleonie

Unbinned likelihood functions
"""

import numpy as np
from scipy import optimize

# We do not need to bin our energies in the unbinned approach,
# we only need to evaluate the respective pdf at the meassured 
# energy values. To do this we calculate N_tot and dRdE once

def diffrecoil(target,m_dm,g_l,g_heff,bg,bl,E0,E1,E2):
    """ Gives the differential recoil rate for a bi-portal model for the recoil
        energy values in a certain sample via the respective integrands E0,E1,E2.
        Returns a 1-dim array, with a value for dR/dE for each energy in the sample.
        Resolution and exposure are already considered here.
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    g_l : float
        Light mediator coupling (dimensionless).
    g_heff : float
        Effective coupling of the heavy mediator in MeV^-2
    bg : array 
        Array of background contribution for each energy value.
    bl : float
        Background level.
    E0,E1,E2 : array
        Array of contributions to the differential event rate as returned by the function 
        integrand0(), integrand1(), integrand2() in statistics, respectively.
    """ 
    dRdE = 0 
    for j in range(0,target.nelements):
        dRdE += target.prefactor(m_dm,j)*(g_heff**2*E0[j,:]
                +2*(g_heff*g_l)*E1[j,:]
                +g_l**2*E2[j,:])
    if target.background != 0:
        dRdE += bg*bl
    return dRdE*target.exposure

def Ntot(target,m_dm,m_Z_l,g_l,g_heff,ibg,bl,i0,i1,i2):
    """ Gives the total number of expected events (float) for a bi-portal model for the recoil
        energy values in a certain sample via the respective integrals i0,i1,i2.
        Resolution and exposure are already considered here.
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    g_l : float
        Light mediator coupling (dimensionless).
    g_heff : float
        Effective coupling of the heavy mediator in MeV^-2
    ibg : float
        Background contribution to the total number of events.
    bl : float
        Background level.
    i0,i1,i2 : array
        Array of contributions to the differential event rate as returned by the function 
        integral0(), integral1(), integral2() in statistics, respectively.
    """ 
    N_tot = 0 
    for j in range(0,target.nelements):
        N_tot += target.prefactor(m_dm,j)*(g_heff**2*i0[j]
                +2*(g_heff*g_l)*i1[j]
                +g_l**2*i2[j])
    if target.background != 0:
        N_tot += ibg*bl
    N_tot = (N_tot)*target.exposure
    return N_tot



def llgl(param,bg,E0,E1,E2,ibg,i0,i1,i2,target,m_dm,m_Z_l,g_heff,g_order):
    """ Returns negative log-likelihood function in g_l and the background level bl
        for all allowed values of g_l
    ----------
    param : array
        Parameters in which we the function can be minimized, param[0]=bl, param[1]=g_l
    bg : array 
        Array of background contribution for each energy value.
    E0,E1,E2 : array
        Array of contributions to the differential event rate as returned by the function 
        integrand0(), integrand1(), integrand2() in statistics, respectively.
    ibg : float
        Background contribution to the total number of events.
    i0,i1,i2 : array
        Array of contributions to the differential event rate as returned by the function 
        integral0(), integral1(), integral2() in statistics, respectively.
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    g_heff : float
        Effective coupling of the heavy mediator in MeV^-2.
    g_order : float
        Estimate of the order of the light mediator coupling, can be calculated with the
        statistics.getglorder() function or guessed. 
    """
    bl = param[0]
    g_l = param[1]*g_order
    Ntot_model = Ntot(target,m_dm,m_Z_l,g_l,g_heff,ibg,bl,i0,i1,i2)
    ll = -Ntot_model
    dRdE = diffrecoil(target,m_dm,g_l,g_heff,bg,bl,E0,E1,E2)
    for x in dRdE:
        ll += np.log(x)
        if np.isfinite(ll) == False:
            ll = float('-inf') #float('inf') 
            break 
    return -ll

def llglpos(param,bg,E0,E1,E2,ibg,i0,i1,i2,target,m_dm,m_Z_l,g_heff,g_order):
    """ Returns negative log-likelihood function in g_l and the background level bl
        if only postive values of g_l are allowed. 
    ----------
    param : array
        Parameters in which we the function can be minimized, param[0]=bl, param[1]=g_l
    bg : array 
        Array of background contribution for each energy value.
    E0,E1,E2 : array
        Array of contributions to the differential event rate as returned by the function 
        integrand0(), integrand1(), integrand2() in statistics, respectively.
    ibg : float
        Background contribution to the total number of events.
    i0,i1,i2 : array
        Array of contributions to the differential event rate as returned by the function 
        integral0(), integral1(), integral2() in statistics, respectively.
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    g_heff : float
        Effective coupling of the heavy mediator in MeV^-2.
    g_order : float
        Estimate of the order of the light mediator coupling, can be calculated with the
        statistics.getglorder() function or guessed. 
    """
    bl = param[0]
    g_l = param[1]*g_order
    if g_l<0:
        ll = float('-inf')
    else:
        Ntot_model = Ntot(target,m_dm,m_Z_l,g_l,g_heff,ibg,bl,i0,i1,i2)
        ll = -Ntot_model
        dRdE = diffrecoil(target,m_dm,g_l,g_heff,bg,bl,E0,E1,E2)
        for x in dRdE:
            ll += np.log(x)
            if np.isfinite(ll) == False:
                ll = float('-inf') #float('inf') 
                break
    return -ll


def llbl(param,bg,E0,E1,E2,ibg,i0,i1,i2,target,m_dm,m_Z_l,g_l,g_heff,g_order):
    """ Returns negative log-likelihood function in the background level bl.
    ----------
    param : float
        Parameter in which we the function can be minimized, param=bl.
    bg : array 
        Array of background contribution for each energy value.
    E0,E1,E2 : array
        Array of contributions to the differential event rate as returned by the function 
        integrand0(), integrand1(), integrand2() in statistics, respectively.
    ibg : float
        Background contribution to the total number of events.
    i0,i1,i2 : array
        Array of contributions to the differential event rate as returned by the function 
        integral0(), integral1(), integral2() in statistics, respectively.
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    g_l : float
        Light mediator coupling as given by the minimization, i.e. full coupling is
        g_l*g_order.
    g_heff : float
        Effective coupling of the heavy mediator in MeV^-2.
    g_order : float
        Estimate of the order of the light mediator coupling, can be calculated with the
        statistics.getglorder() function or guessed. 
    """
    bl = param
    Ntot_model = Ntot(target,m_dm,m_Z_l,g_l*g_order,g_heff,ibg,bl,i0,i1,i2)
    ll = -Ntot_model
    dRdE = diffrecoil(target,m_dm,g_l*g_order,g_heff,bg,bl,E0,E1,E2)
    for x in dRdE:
        ll += np.log(x)
        if np.isfinite(ll) == False:
            ll = float('-inf') #float('inf') 
            break
    return -ll


def ts_excl_gl(g_l,bg,E0,E1,E2,ibg,i0,i1,i2,target,denom,bestN,testexcl,m_dm,m_Z_l,g_heff,g_order):
    """ Profile likelihood ratio test statistic in g_l. Profiling over the background level.
    ----------
    g_l : float
        Light mediator coupling as given by the minimization, i.e. full coupling is
        g_l*g_order.
    bg : array 
        Array of background contribution for each energy value.
    E0,E1,E2 : array
        Array of contributions to the differential event rate as returned by the function 
        integrand0(), integrand1(), integrand2() in statistics, respectively.
    ibg : float
        Background contribution to the total number of events.
    i0,i1,i2 : array
        Array of contributions to the differential event rate as returned by the function 
        integral0(), integral1(), integral2() in statistics, respectively.
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    denom : float
        Denominator of the likelihood ratio, i.e. value of the likelihood function 
        with best estimate in both g_l and all nuissance parameters (here bl).
    bestN : 
        Total number of signal events predicted by the best ML fit parameters.
    testexcl : 
        Value from quantile of chi-distribution for the respective p-value.
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    g_heff : float
        Effective coupling of the heavy mediator in MeV^-2.
    g_order : float
        Estimate of the order of the light mediator coupling, can be calculated with the
        statistics.getglorder() function or guessed. 
    """    
    N = Ntot(target,m_dm,m_Z_l,g_l*g_order,g_heff,0*ibg,1,i0,i1,i2)
    if bestN>=N:
        res = 0
    else:
        profile = optimize.minimize(llbl,1,args=(bg,E0,E1,E2,ibg,i0,i1,i2,target,m_dm,m_Z_l,g_l,g_heff,g_order))
        nom = profile.fun
        #nom = llbl(1,bg,E0,E1,E2,ibg,i0,i1,i2,target,m_dm,m_Z_l,g_l,g_heff,g_order)
        res = 2*(nom-denom)
    if res <= testexcl:
        res = 0
    return res
