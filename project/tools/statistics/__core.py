# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

"""
This is a private module which is only ever called over init, or directly within binned.py or unbinned.py. To
access any of the function below, please do so by calling 'statistics'
"""

import numpy as np
from scipy import stats
import project.tools.formfactor as ff
import project.tools.velocity_int as vi
from project.constants import atomic, darkmatter, conversion, cosmo
from project.tools.convolution import convolve
from project.tools.convolution import gaussian
from project.tools import magnitude

"""Target definition"""
#for every experiment a class 'target' has to be passed, with '*args'
#denoting the mass number and atomic number of each element in the target 
#molecule and the number of these elements, ie. [A_O,Z_O,4] for O in CaWO4,
# in addition we also pass experiment specific keyword arguments like exposure,
# threshold and the sigma of a gaussian 

class target:
    def __init__(self, *args, exposure, thr, sigma, bg, **kwargs):
        """ For every target material a class 'target' has to be passed, under whichs namespace
        one can then find various functions which are needed in order to calculate likelihoods. 
        Attributes
        ----------
        *args : tuple
            Denoting the mass number of each element in the target molecule and 
            the number of these elements, e.g. [A_Ca,1],[A_W,1],[A_O,4] for CaWO4.
        exposure : float 
            Exposure of experiment in kgd, if you do not want to specify it pass 1.
        thr : float 
            Threshold of the experiment in keV, if you do not want to specify it pass a very 
            low number, i.e. 1e-5.
        sigma : float
            Width of the Gaussian resolution of the experiment in keV, if you do not want to
            specify it pass 0.
        bg : tuple 
            Background model of the experiment, pass either a single value in 1/keVkgday for a
            flat background or two values [k,d] for a linear background of dR_b/dE = k*E+d.
        roi: float (optional)
            Specify maximum value of region of interest in keV, minimum value is given by 
            the threshold. Has to be passed for flat background models!
        """

        self.A = []
        self.mass = []
        self.frac = []
        self.nelements = len(args)
        self.totmass = 0
        self.exposure = exposure
        self.threshold = thr
        # sigma is for the convolution with an energy resolution, if sigma=0
        # no resolution is considered
        self.sigma = sigma
        self.background = bg
        self.res = True
        if sigma == 0:
            self.sigma = 0.1
            self.res = False
        else:
            self.sigma = sigma
            self.res = True
        # we define a region of interest, if not given            
        if "roi" in kwargs:
            self.roimax = kwargs.get("roi")
        else: 
            if len(bg) == 2:
                self.roimax = -bg[1]/bg[0]
            elif len(bg) == 1:
                raise KeyError('for constant bg a ROI is needed')
            else:
                raise KeyError('some wrong background input')
        roinum = int((self.roimax-self.threshold)/self.sigma)
        self.roi = np.linspace(self.threshold,self.roimax,roinum)
        # here we produce the E' for the resolution integration
        Enum = int((self.roimax-1e-5)/self.sigma)
        self.Eprime = np.linspace(1e-5,self.roimax,Enum)
        for element in args:
            self.A.append(element[0])
            self.mass.append(element[0]*atomic.amu)
            self.totmass += element[0]*element[1]*atomic.amu
        for i,element in zip(range(0,self.nelements),args):
            self.frac.append((self.mass[i]*element[1])/self.totmass)
        return 
    
    def initialize_vdf(self, **kwargs):
        """ To initialize the VDF required for velocity integration. 
        ----------
        I_index: Integer. Which velocity integration to use. see velocity_int.py
                 default 0: This represents SHM, Maxwell-Botzmann distribution 
                 with Vcir = 220 km/s, Vesc = 544 km/s
                 I_index = 1 if custom vesc and vcir has to be passed to SHM. 
                 I_index = 2 if custom VDF (and V) has to be passed, custom vesc and vcir also
                           can be passed here.
        VDF, V: Velocity distribution function (VDF) at velocity in earth reference frame (V),
                should be passed iff I_index = 2.
        vesc: Use this keyword if a different value for local escape velocity from 
              the default value of 544 has to be passed.
        vcir: Use this keyword if a different vlaue for the circular speed of local 
              standard of rest has to be used (default = 220 km/s). 
        vearth: If the keyword vcir is not used, a custom value for the velocity of
                earth with respect to the galactic rest frame can be passed.
        vrms: Similarly, if vcir is not used, a custom value for the vrms velocity 
              can be passed. Default value is sqrt(3/2)vcir
        """
        if 'I_index' in kwargs:
            self.I_index = kwargs.get('I_index')
        else:
            self.I_index = 0
        if self.I_index == 2:
            self.VDF = kwargs.get('VDF')
            self.V = kwargs.get('V')
            
        if "vesc" in kwargs:
            self.vesc = kwargs.get("vesc")
        else:
            self.vesc = cosmo.v_esc
            
        if 'vcir' in kwargs:
            self.vcir = kwargs.get('vcir')
            self.vearth = self.vcir*(1.05 + 0.07)
            self.vrms = np.sqrt(3./2.)*self.vcir
        else: 
            if "vearth" in kwargs:
                self.vearth = kwargs.get("vearth")
            else:
                self.vearth = cosmo.v_earth
        
            if "vrms" in kwargs:
                self.vrms = kwargs.get("vrms")
            else:
                self.vrms = cosmo.w
            
    
# the following functions are needed to compute the differential recoil spectrum
# or any related quantities such as pdfs 

    def prefactor(self,m_dm,index,rh=darkmatter.rho_wimp):
        """ Prefactor before velocity integral in differential event rate.
        ----------
        m_dm : float
            DM mass in GeV.
        index : integer
            Index of the element component in the target mocecule, e.g. 0 for Na in NaI.
        rh : dark matter local density. Default value is 0.3 GeV/cm^3. 
        """
        return (((rh*self.A[index]**2*9)/(2*np.pi*m_dm))*
                    self.frac[index]*conversion.conv_g)

    
    def vF(self,m_dm,E,index):
        """ Velocity integral in differential event rate.
        ----------
        m_dm : float
            DM mass in GeV.
        E : array
            Energy array in keV.
        index : integer
            Index of the element component in the target mocecule, e.g. 0 for Na in NaI.
        """       
        if self.I_index == 0:
            ans = vi.I(vi.v_min(self.mass[index],m_dm,E))*ff.F2(self.A[index],self.mass[index],E)
        elif self.I_index == 1:
            ans = vi.I1(vi.v_min(self.mass[index],m_dm,E),vesc = self.vesc, vrms = self.vrms,
            vearth = self.vearth) * ff.F2(self.A[index], self.mass[index],E)
        elif self.I_index == 2:
            ans = vi.I2(self.V, self.VDF, vi.v_min(self.mass[index],m_dm,E), self.vesc, 
            self.vearth)
        elif self.I_index == 'AM':
            ans = vi.AM(vi.v_min(self.mass[index],m_dm,E))*ff.F2(self.A[index],self.mass[index],E)
        return ans
        
    def mus_N(self,m_dm):
        """ Calculates the reduced DM nucleus mass for smallest nucleus in the target.
            This is needed later for max. recoil energy
        ----------
        m_dm : float
            DM mass in GeV.
        """ 
        minN = min(self.mass)
        muN = (m_dm*minN)/(m_dm+minN)
        return muN
    
    def E_max(self,m_dm): #maximum recoil energy in keV
        """ Calculates maximum recoil energy in keV which can be achieved for a DM particle 
            of certain mass m_dm.
        ----------
        m_dm : float
            DM mass in GeV.
        """ 
        minN = min(self.mass)
        #Emax = (2*self.mus_N(m_dm)**2*(cosmo.v_esc+cosmo.v_earth)**2)/minN 
        Emax = (2*self.mus_N(m_dm)**2*(self.vesc+self.vearth)**2)/minN
        Emax = (Emax*10**6)/(cosmo.c)**2
        return Emax
    

# integrand 0-2 are the energy dependent terms of the recoil spectrum, split up 
# in such a way that couplings can be seperated from them
# if sigma != 0 they are convoluted with a guassian energy resolution
# the 'index' parameter stands for the index of each element within the target 
# molecule

    def integrand0(self,m_dm,E,index):
        """ Gives the energy-dependent contribution to the differential recoil rate for the heavy
            mediator (i.e. no dependence on m_Zl) for an element in the target of
            a certain index.
        Attributes
        ----------
        m_dm : float
            DM mass in GeV.
        E : array
            Energy array in keV.
        index : integer
            Index of the element in the target molecule as given in the definition
            of the namespace, e.g. index 0 for Na in NaI
        """ 
        if self.res == False:
            result = self.vF(m_dm,E,index)
        else:
            rate = self.vF(m_dm,self.Eprime,index)
            result = convolve(self.sigma,E,self.Eprime,rate)
        return result
    
    def integrand1(self,m_dm,m_Z_l,E,index):
        """ Gives the energy-dependent contribution to the differential recoil rate for the interference
            term for an element in the target of a certain index.
        Attributes
        ----------
        m_dm : float
            DM mass in GeV.
        m_Z_l : float
            Light mediator mass in MeV.
        E : array
            Energy array in keV.
        index : integer
            Index of the element in the target molecule as given in the definition
            of the namespace, e.g. index 0 for Na in NaI
        """ 
        if self.res == False:
            result = self.vF(m_dm,E,index)/(2*self.mass[index]*E+m_Z_l**2)
        else:
            rate = self.vF(m_dm,self.Eprime,index)/(2*self.mass[index]*self.Eprime+m_Z_l**2)
            result = convolve(self.sigma,E,self.Eprime,rate)
        return result
        
    def integrand2(self,m_dm,m_Z_l,E,index):
        """ Gives the energy-dependent contribution to the differential recoil rate for the light mediator
            term for an element in the target of a certain index.
        Attributes
        ----------
        m_dm : float
            DM mass in GeV.
        m_Z_l : float
            Light mediator mass in MeV.
        E : array
            Energy array in keV.
        index : integer
            Index of the element in the target molecule as given in the definition
            of the namespace, e.g. index 0 for Na in NaI
        """ 
        if self.res == False:
            result = self.vF(m_dm,E,index)/(2*self.mass[index]*E+m_Z_l**2)**2
        else:
            rate = self.vF(m_dm,self.Eprime,index)/(2*self.mass[index]*self.Eprime+m_Z_l**2)**2
            result = convolve(self.sigma,E,self.Eprime,rate)
        return result
    
    def bg(self,E):
        """ Gives the energy contribution to the differential recoil rate from the background.
        ----------
        E : array
            Energy array in keV.
        """ 
        if len(self.background)>1:
            result = E*self.background[0]+self.background[1]
        else:
            result = np.ones(np.shape(E))*self.background[0]
        #K_peak = gaussian(self.sigma,E,3.) # a gaussian peak at 3 keV
        #result += K_peak*result*0.25
        return result
    

# the following terms are integrations over the energy dependent terms which 
# are needed to calculate the the total number integral or the integrals
# over the bins for a binned likelihood, the results are 2-dim arrays
# with one axis nelements and the other the size of the E-array

    def integral0(self,m_dm,E):
        """ Calculates the integral of integrand0 for a certain energy array.
        """ 
        int0 = np.zeros(self.nelements)
        for j in range(0,self.nelements):
            int0[j] = np.trapz(self.integrand0(m_dm,E,j),E)
        return int0
    
    def integral1(self,m_dm,m_Z_l,E):
        """ Calculates the integral of integrand1 for a certain energy array.
        """ 
        int1 = np.zeros(self.nelements)
        for j in range(0,self.nelements):
            int1[j] = np.trapz(self.integrand1(m_dm,m_Z_l,E,j),E)
        return int1

    def integral2(self,m_dm,m_Z_l,E):
        """ Calculates the integral of integrand2 for a certain energy array.
        """ 
        int2 = np.zeros(self.nelements)
        for j in range(0,self.nelements):
            int2[j] = np.trapz(self.integrand2(m_dm,m_Z_l,E,j),E)
        return int2
    
    def bgintegral(self,E):
        """ Calculates the integral over the background function for a certain energy array.
        """ 
        return np.trapz(self.bg(E),E)
    
    #EXP-new
    def expbgintegral(self,Exp0,Expt,E):
    	def expbg(E):
    		result = Exp0*(np.exp(-(E-self.threshold)/Expt))
    		return result
    	return np.trapz(expbg(E),E)

# full differential recoil spectrum:
    def diffrecoil(self,m_dm,m_Z_l,g_l,g_heff,E):
        """ Gives the differential recoil rate for the bi-portal model for a certain
            energy array. Resolution and exposure are already considered here.
        Attributes
        ----------
        m_dm : float
            DM mass in GeV.
        m_Z_l : float
            Light mediator mass in MeV.
        g_l : float
            Light mediator coupling (dimensionless).
        g_heff : float
            Effective coupling of the heavy mediator in MeV^-2
        E : array
            Energy array in keV
        """ 
        dRdE = 0 
        for j in range(0,self.nelements):
            dRdE += self.prefactor(m_dm,j)*(g_heff**2*self.integrand0(m_dm,E,j)
                    +2*(g_heff*g_l)*self.integrand1(m_dm,m_Z_l,E,j)
                    +g_l**2*self.integrand2(m_dm,m_Z_l,E,j))
        if self.background != 0:
            dRdE += self.bg(E)
        return dRdE*self.exposure

    def Ntotsimp(self,m_dm,m_Z_l,g_l,g_heff):
        """ Gives the total number of events for the bi-portal model. 
            Resolution, threshold (ROI) and exposure are already considered here.
        Attributes
        ----------
        m_dm : float
            DM mass in GeV.
        m_Z_l : float
            Light mediator mass in MeV.
        g_l : float
            Light mediator coupling (dimensionless).
        g_heff : float
            Effective coupling of the heavy mediator in MeV^-2.
        """ 
        recoil = self.diffrecoil(m_dm,m_Z_l,g_l,g_heff,self.roi)
        N_tot = np.trapz(recoil,self.roi)
        return N_tot

    def pdfsimp(self,m_dm,m_Z_l,g_l,g_heff,E):
        """ Gives the normalized differential recoil rate (pdf) for the bi-portal models for a certain
            energy array. Resolution, threshold and exposure are already considered here.
        Attributes
        ----------
        m_dm : float
            DM mass in GeV.
        m_Z_l : float
            Light mediator mass in MeV.
        g_l : float
            Light mediator coupling (dimensionless).
        g_heff : float
            Effective coupling of the heavy mediator in MeV^-2.
        E : array
            Energy array in keV.
        """ 
        recoil = self.diffrecoil(m_dm,m_Z_l,g_l,g_heff,E)
        N_tot = self.Ntotsimp(m_dm,m_Z_l,g_l,g_heff)
        pdf = recoil/N_tot
        return N_tot, pdf

    def pdfbg(self,E):  
        """ Gives the normalized differential recoil rate (pdf) for a background for a certain
            energy array in keV.
        Attributes
        ----------
        E : array
            Energy array in keV.
        """   
        bg = self.bg(E)*self.exposure
        N_tot_bg = np.trapz(bg, E)
        pdf = bg/N_tot_bg
        return N_tot_bg, pdf

         
"""Poisson distribution"""
# pdf for a poisson distribution lambd=lambda, for high values of lambda or x
# we approximate the poisson distribution with a Gaussian/Normal distribution
# for negative and thus unphysical values of lambda, we pass a very small
# value for the distribution  

def poisson(x,lambd):
    if lambd<100 and lambd>0 and x<100:
        pois = (lambd**x/np.math.factorial(x))*np.exp(-lambd)
    elif lambd<=0:
        pois = 1e-15
    else:
        mu = lambd
        sigma = np.sqrt(lambd)
        pois = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu)/sigma)**2)
    if pois<=0:
        pois = 1e-15
    return pois



def mocksample(target,m_dm,m_Z_l,g_l,g_heff,seed = None):
    """ Inverse cdf mock sample generation, with a DM signal.
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
    """ 
    Enum = int(1e3)
    E_array = np.linspace(target.threshold,target.roi[-1],Enum)
    Ntot, pdf = target.pdfsimp(m_dm,m_Z_l,g_l,g_heff,E_array)
    # calculate the cdf for the energy array set above
    cdf = np.zeros(int(Enum))
    for i in range(int(Enum)-1):
        dcdf = np.trapz(pdf[i:i+2],E_array[i:i+2])
        if i == 0:
            cdf[i] = dcdf
        else:
            cdf[i] = cdf[i-1] + dcdf
    cdf[-1]=cdf[-2]

    # draw a random number for the total number of events
    #Ntot_rand = int(np.floor(np.random.poisson(Ntot,1)))
    Ntot_rand = int(Ntot)
    
    # draw the mocksample
    if seed:
        np.random.seed(seed)
    Esample = []
    for u in np.random.uniform(0,1,size=Ntot_rand):
        index = (np.abs(cdf-u)).argmin()
        Esample.append(E_array[index])
    sample = np.array(Esample)

    return {'sample': sample, 'Nrand': Ntot_rand, 'cdf':cdf, 'E':E_array}


def bgsample(target,seed = None):
    """ Inverse cdf mock background sample generation, without a DM signal.
    Attributes
    ----------
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    """
    if np.shape(target.roi)[0]<1e3:
        roinum = int(1e3)
        E_array = np.linspace(target.threshold,target.roimax,roinum)
    else:
        E_array = target.roi
    Ntot, pdf = target.pdfbg(E_array)
    Enum = np.size(E_array)
    # calculate the cdf for the energy array set above
    cdf = np.zeros(int(Enum))
    for i in range(int(Enum)-1):
        dcdf = np.trapz(pdf[i:i+2],E_array[i:i+2])
        if i == 0:
            cdf[i] = dcdf
        else:
            cdf[i] = cdf[i-1] + dcdf
    cdf[-1]=cdf[-2]

    # draw a random number for the total number of events
    #Ntot_rand = int(np.floor(np.random.poisson(Ntot,1)))
    Ntot_rand = int(Ntot)
    
    # draw the mocksample
    if seed:
        np.random.seed(seed)
    Esample = []
    for u in np.random.uniform(0,1,size=Ntot_rand):
        index = (np.abs(cdf-u)).argmin()
        Esample.append(E_array[index])
    sample = np.array(Esample)

    return {'sample': sample, 'Nrand': Ntot_rand, 'cdf':cdf, 'E':E_array}


# When we want to find the ML estimators of the couplings we first have to estimate the order of
# the couplings, so the minimization does not have to work with very small numbers. This first 
# estimation is based only on the total number of events and is evavluated by comparing the 
# total number in the sample to the total number of events achieved in a heavy limit model
# with coupling 1e-10 MeV^-2

def getgorder(N_sample,target,m_dm):
    """ Function to estimate the order of the coupling to simplify the minimization 
        procedure.
    ----------
    N_sample : integer
        Number of recoil events in the sample.
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    """ 
    g_heff = 1e-10
    N_tot = target.Ntotsimp(m_dm,10,0,g_heff)
    ratio = N_sample/N_tot
    g_heff = ratio**(0.5)*g_heff
    return magnitude(g_heff)

def getglorder(N_sample,target,m_dm,m_Z_l):
    """ Function to estimate the order of the coupling to simplify the minimization 
        procedure.
    ----------
    N_sample : integer
        Number of recoil events in the sample.
    target : class
        Namespace (class) of the target used as given by the statistics.target
        class (see above).
    m_dm : float
        DM mass in GeV.
    m_Z_l : float
        Light mediator mass in MeV.
    """ 
    g_l = 1e-10
    N_tot = target.Ntotsimp(m_dm,m_Z_l,g_l,0)
    ratio = N_sample/N_tot
    g_l = ratio**(0.5)*g_l
    return magnitude(g_l)



