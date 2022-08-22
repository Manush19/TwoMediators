# -*- coding: utf-8 -*-
"""
@author: einfaltleonie

Binned likelihood functions
"""

import numpy as np
from scipy import optimize
import sys
sys.path.append('..')
sys.path.append('../..')
sys.path.append('../../..')
sys.path.append('../../../..')
sys.path.append('../../../../..')
import project.quadrantHopping as quadH
from iminuit import Minuit


def binsample(sample,binnum,accuracy):
    """ This function bins a sample for a certain number of bins, saves the limit values of 
    the bins and then generates energy arrays for each bin with a certain accuracy.
    This is important for the later integration over the bins. A good choice of parameters is
    is e.g. a bin-width of 5x the resolution and an accuracy of 10, then the energy values are spread 
    with a distance of half the resolution. This function returns a dictionary.
    Attributes
    ----------
    sample : array
        Sample of energy values.
    binnum : integer
        Number of bins.
    accuracy : integer
        Number of values in the energy array of each bin.

    Returns: dictionary
    ----------
    binnedsample : histogram
        The binned sample as a numpy histogram object.
    binnedE : array 
        (binnum x accuracy) sized array of energy values for integration.
    Ntot : integer
        Number of elements in the sample.
    binnum : integer
        Number of bins.
    accuracy : integer
        Number of values in the energy array of each bin.
    """
    binnedsample = np.histogram(sample, binnum)
    E_array = []
    for i in range(binnum):
        E_array.extend(np.linspace(binnedsample[1][i],binnedsample[1][i+1],accuracy))
    binnedE = np.array(E_array) 
    Ntot = np.size(sample)   # the size of the sample
    return {
        'binnedsample': binnedsample, 
        'binnedE': binnedE, 
        'Ntot': Ntot,
        'binnum': binnum,
        'accuracy':accuracy}


# We caclulate the three different energy dependent integrals for each bin,
# which are necessary to calculate the binned pdf.
# Each bint is a 2-dim array, where the first index is for the bin number 
# and the second index for the element in the target.

def bint0(target,bsample,m_dm):
    """ Returns a 2-dim array, with the heavy mediator only contribution to the total
        number of expected events in each bin for a certain DM interaction model. 
        The first axis of the array counts the bin-number, the second the index 
        of the element in the target material. 
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample().
    m_dm : float
        DM mass in GeV.
    """
    binnedE = bsample['binnedE']
    accuracy = bsample['accuracy']
    binnum = bsample['binnum']
    int0 = np.zeros([binnum,target.nelements])
    for i in range(binnum):
        int0[i,:] = target.integral0(m_dm,binnedE[int(i*accuracy):int(i*accuracy+accuracy)])
    return int0

def bint1(target,bsample,m_dm,m_Z_l):
    """ Returns a 2-dim array, with the interference contribution to the total
        number of expected events in each bin for a certain DM interaction model. 
        The first axis of the array counts the bin-number, the second the index 
        of the element in the target material. 
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample()
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    """
    binnedE = bsample['binnedE']
    accuracy = bsample['accuracy']
    binnum = bsample['binnum']
    int1 = np.zeros([binnum,target.nelements])
    for i in range(binnum):
        int1[i,:] = target.integral1(m_dm,m_Z_l,binnedE[int(i*accuracy):int(i*accuracy+accuracy)])
    return int1

def bint2(target,bsample,m_dm,m_Z_l):
    """ Returns a 2-dim array, with the light mediator contribution to the total
        number of expected events in each bin for a certain DM interaction model. 
        The first axis of the array counts the bin-number, the second the index 
        of the element in the target material. 
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample()
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    """
    binnedE = bsample['binnedE']
    accuracy = bsample['accuracy']
    binnum = bsample['binnum']
    int2 = np.zeros([binnum,target.nelements])
    for i in range(binnum):
        int2[i,:] = target.integral2(m_dm,m_Z_l,binnedE[int(i*accuracy):int(i*accuracy+accuracy)])
    return int2

def bintbg(target,bsample):
    """ Returns a 2-dim array, with the background contribution to the total
        number of expected events in each bin for a certain DM interaction model. 
        The first axis of the array counts the bin-number, the second the index 
        of the element in the target material. 
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample()
    """
    binnedE = bsample['binnedE']
    accuracy = bsample['accuracy']
    binnum = bsample['binnum']
    intbg = np.zeros(binnum)
    for i in range(binnum):
        intbg[i] = target.bgintegral(binnedE[int(i*accuracy):int(i*accuracy+accuracy)])
    return intbg

# EXP-new    
def bintexpbg(target,bsample,E0,Et):
	binnedE = bsample['binnedE']
	accuracy = bsample['accuracy']
	binnum = bsample['binnum']
	intbg = np.zeros(binnum)
	for i in range(binnum):
		intbg[i] = target.expbgintegral(E0,Et,binnedE[int(i*accuracy):int(i*accuracy+accuracy)])
	return intbg


def bintot(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,bl):
    """ Returns a 1-dim array, with thehypothized number of events in each energy bin. 
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
        Effective coupling of the heavy mediator in MeV^-2.
    b0,b1,b2 : arra
        Array of contributions to the number of energy values as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
    bl: float
        Background level, will be profiled over. 
    """
    bin_tot = np.zeros(np.shape(b0[:,0]))
    for i in range(0,np.shape(b0)[0]):
        for j in range(target.nelements): #summing over elements
            bin_tot[i] += target.prefactor(m_dm,j)*(g_heff**2*b0[i,j]
                     +2*(g_heff*g_l)*b1[i,j]
                     +g_l**2*b2[i,j])
        if target.background != 0:
            bin_tot[i] += bbg[i]*bl
    bin_tot = target.exposure*bin_tot
    return np.array(bin_tot)
    
# EXP-new
def bintotexp(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,bl,expbbg):
    """ Returns a 1-dim array, with thehypothized number of events in each energy bin
    for an exponential background.
    """
    bin_tot = np.zeros(np.shape(b0[:,0]))
    for i in range(0,np.shape(b0)[0]):
        for j in range(target.nelements): #summing over elements
            bin_tot[i] += target.prefactor(m_dm,j)*(g_heff**2*b0[i,j]
                     +2*(g_heff*g_l)*b1[i,j]
                     +g_l**2*b2[i,j])
        if target.background != 0:
            bin_tot[i] += bbg[i]*bl
        bin_tot[i] += expbbg[i]
    bin_tot = target.exposure*bin_tot
    return np.array(bin_tot)

def bintotrho(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,bl,rh):
    """ Returns a 1-dim array, with thehypothized number of events in each energy bin
    including the local dark matter density (rh) as a free parameter.
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
        Effective coupling of the heavy mediator in MeV^-2.
    b0,b1,b2 : arra
        Array of contributions to the number of energy values as returned by the function
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array
        Array of background contribution in each bin.
    bl: float
        Background level, will be profiled over.
    rh: float
        local dark matter density. 
    """
    bin_tot = np.zeros(np.shape(b0[:,0]))
    for i in range(0,np.shape(b0)[0]):
        for j in range(target.nelements): #summing over elements
            bin_tot[i] += target.prefactor(m_dm,j,rh)*(g_heff**2*b0[i,j]
                     +2*(g_heff*g_l)*b1[i,j]
                     +g_l**2*b2[i,j])
        if target.background != 0:
            bin_tot[i] += bbg[i]*bl
    bin_tot = target.exposure*bin_tot
    return np.array(bin_tot)

         
def bintot2(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,bl,Z = [0,0],theta_l = np.pi/4.,theta_h = np.pi/4.):
    """ Returns a 1-dim array, with thehypothized number of events in each energy bin. 
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
        Effective coupling of the heavy mediator in MeV^-2.
    b0,b1,b2 : arra
        Array of contributions to the number of energy values as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
    bl: float
        Background level, will be profiled over. 
    """
    
    
    
    bin_tot = np.zeros(np.shape(b0[:,0]))
    for i in range(0,np.shape(b0)[0]):
        for j in range(target.nelements): #summing over elements
            pH = np.sqrt(2)*(Z[j]*np.cos(theta_h)+(target.A[j]-Z[j])*np.sin(theta_h))/target.A[j]
            pL = np.sqrt(2)*(Z[j]*np.cos(theta_l)+(target.A[j]-Z[j])*np.sin(theta_l))/target.A[j]
    		
            bin_tot[i] += target.prefactor(m_dm,j)*(g_heff**2*b0[i,j]*pH**2
                     +2*(g_heff*g_l)*b1[i,j]*pH*pL
                     +g_l**2*b2[i,j])*pL**2
        if target.background != 0:
            bin_tot[i] += bbg[i]*bl
    bin_tot = target.exposure*bin_tot
    return np.array(bin_tot)
    
def bintot3(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,sl,bl,Z = [0,0],theta_l = np.pi/4.,theta_h = np.pi/4.,E=None):
    """ Returns a 1-dim array, with thehypothized number of events in each energy bin. 
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
        Effective coupling of the heavy mediator in MeV^-2.
    b0,b1,b2 : arra
        Array of contributions to the number of energy values as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
    bl: float
        Background level, will be profiled over. 
    """
    bin_tot = np.zeros(np.shape(b0[:,0]))
    for i in range(0,np.shape(b0)[0]):
        for j in range(target.nelements): #summing over elements
            pH = np.sqrt(2)*(Z[j]*np.cos(theta_h)+(target.A[j]-Z[j])*np.sin(theta_h))/target.A[j]
            pL = np.sqrt(2)*(Z[j]*np.cos(theta_l)+(target.A[j]-Z[j])*np.sin(theta_l))/target.A[j]
    		
            bin_tot[i] += target.prefactor(m_dm,j)*(g_heff**2*b0[i,j]*pH**2
                     +2*(g_heff*g_l)*b1[i,j]*pH*pL
                     +g_l**2*b2[i,j])*pL**2
        if target.background != 0:
            bin_tot[i] += bbg[i]*(sl*E[i]+bl)
    bin_tot = target.exposure*bin_tot
    return np.array(bin_tot)    

def llgl(param,bsample,b0,b1,b2,bbg,target,m_dm,m_Z_l,g_heff,g_order):
    """ Returns negative log-likelihood function in g_l and the background level bl
        for all allowed values of g_l
    ----------
    param : array
        Parameters in which we the function can be minimized, param[0]=bl, param[1]=g_l
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample().
    b0,b1,b2 : array
        Array of contributions to the number of energy values in each bin as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
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
    binnedsample = bsample['binnedsample']
    binnum = bsample['binnum']
    bl = param[0]
    g_l = param[1]*g_order
    lambd_array = bintot(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,bl)
    ll =  -np.sum(lambd_array)
    for i in range(binnum):
        lambd_i = lambd_array[i]
        n_i = binnedsample[0][i]
        ll += n_i*np.log(lambd_i)
        if np.isfinite(ll) == False:
            ll = float('-inf') #float('inf') 
            break
    return -ll


def llglpos(param,bsample,b0,b1,b2,bbg,target,m_dm,m_Z_l,g_heff,g_order):
    """ Returns negative log-likelihood function in g_l and the background level bl
        for only positive allowed values of g_l
    ----------
    param : array
        Parameters in which we the function can be minimized, param[0]=bl, param[1]=g_l
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample().
        b0,b1,b2 : array
        Array of contributions to the total bin number as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
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
    binnedsample = bsample['binnedsample']
    binnum = bsample['binnum']
    bl = param[0]
    g_l = param[1]*g_order
    if g_l<0:
        ll = float('-inf') 
    elif bl < 0:
    	ll = float('-inf')
    else:
        lambd_array = bintot(target,m_dm,m_Z_l,g_l,g_heff,b0,b1,b2,bbg,bl)
        ll =  -np.sum(lambd_array)
        for i in range(binnum):
            lambd_i = lambd_array[i]
            n_i = binnedsample[0][i]
            ll += n_i*np.log(lambd_i)
            if np.isfinite(ll) == False:
                ll = float('-inf') #float('inf') 
                break
    return -ll

def llbl(param,bsample,b0,b1,b2,bbg,target,m_dm,m_Z_l,g_l,g_heff,g_order):
    """ Returns negative log-likelihood function in the background level bl.
    ----------
    param : float
        Parameter in which we the function can be minimized, param=bl.
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample().
        b0,b1,b2 : array
        Array of contributions to the total bin number as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
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
    binnedsample = bsample['binnedsample']
    binnum = bsample['binnum']
    bl = param
    if g_l<0:
        ll = float('-inf') 
    elif bl < 0:
    	ll = float('-inf')
    else:
        lambd_array = bintot(target,m_dm,m_Z_l,g_l*g_order,g_heff,b0,b1,b2,bbg,bl)
        ll =  -np.sum(lambd_array)
        for i in range(binnum):
            lambd_i = lambd_array[i]
            n_i = binnedsample[0][i]
            ll += n_i*np.log(lambd_i)
            if np.isfinite(ll) == False:
                ll = float('-inf') #float('inf') 
                break
    return -ll


def ts_excl_gl(g_l,bsample,b0,b1,b2,bbg,target,denom,bestN,testexcl,m_dm,m_Z_l,g_heff,g_order):
    """ Profile likelihood ratio test statistic in g_l. Profiling over the background level.
    ----------
    g_l : float
        Light mediator coupling as given by the minimization, i.e. full coupling is
        g_l*g_order.
    bsample : dictionary
        A dictionary for the binned sample as returned by binsample().
        b0,b1,b2 : array
        Array of contributions to the total bin number as returned by the function 
        bint0, bint1, bint2 in statistics.binned, respectively.
    bbg : array 
        Array of background contribution in each bin.
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
    
    profile = optimize.minimize(llbl,1,args=(bsample,b0,b1,b2,bbg,target,m_dm,m_Z_l,g_l,g_heff,g_order))
    bl = profile.x
    
    N = np.sum(bintot(target,m_dm,m_Z_l,g_l*g_order,g_heff,b0,b1,b2,0*bbg,bl))
    if bestN>=N:
        res = 0
    else:
        nom = profile.fun
        res = 2*(nom-denom)
    if res <= testexcl:
        res = 0
    return res

