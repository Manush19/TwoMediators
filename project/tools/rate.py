# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np
import project.tools.formfactor as ff
import project.tools.velocity_int as vi
from project.tools.convolution import convolve 
from project.constants import atomic, darkmatter, conversion, cosmo


class target:
    def __init__(self, *args, **kwargs):
        """ For every target material a class 'target' has to be passed, under whichs namespace
        one can then find various recoil spectra options. 
        ----------
        *args : tuple
            Denoting the mass number and atomic number of each element in the target molecule and 
            the number of these elements, e.g. [A_Ca,Z_Ca,1],[A_W,Z_W,1],[A_O,Z_O,4] for CaWO4:
        sigma : float (optional)
            Sigma in keV value for Gaussian detector resolution, default = 0.
        exp : float (optional)
            Exposure of experiment in kgd, default = 1.
        thr : float (optional)
            Threshold of the experiment in keV, default = 1e-5 (~ 0) keV 
        """
        self.A = []
        self.Z = []
        self.mass = []
        self.frac = []
        self.nelements = len(args)
        self.totmass = 0
        for element in args:
            self.A.append(element[0])
            self.Z.append(element[1])
            self.mass.append(element[0]*atomic.amu)
            self.totmass += element[0]*element[2]*atomic.amu
        for i,element in zip(range(0,self.nelements),args):
            self.frac.append((self.mass[i]*element[2])/self.totmass)
        if 'sigma' in kwargs:
            self.sigma = kwargs.get('sigma')
        else:
            self.sigma = 0
        if 'exp' in kwargs:
            self.exp = kwargs.get('exp')
        else:
            self.exp = 1
        if 'thr' in kwargs:
            self.thr = kwargs.get('thr')
        else: 
            self.thr = 1e-4
        if self.sigma == 0:
            self.Enum = int((40-1e-5)/0.05)
        else:
            self.Enum = int((40-1e-5)/self.sigma)
        self.Eprime = np.linspace(1e-5,40,self.Enum)
        return 
             
    
    #reduced DM nucleon mass:        
    def mu_p(self,m_dm):
        mup = (m_dm*atomic.m_p)/(m_dm+atomic.m_p)
        return mup
     
    #reduced DM nucleus mass, needed for min. velocity:        
    def mu_N(self,m_dm):
        muN = []
        for i in range(0,self.nelements):
            muN.append((m_dm*self.mass[i])/(m_dm+self.mass[i]))
        return muN
    

   
    def dRdE_heavy(self,m_dm,sigma_p,E):
        """ Heavy mediator model differential event rate (no mediator mass needed).
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        sigma_p : float
            DM-nucleon cross section in pb.
        E : array
            Array of energies in keV. 
        """

        rate = 0
        if self.sigma == 0:
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*sigma_p)/
                        (2*m_dm*self.mu_p(m_dm)**2))
                rate += (self.frac[i]*prefactor*conversion.conv*
                            vi.I(vi.v_min(self.mass[i],m_dm,E))*
                            ff.F2(self.A[i],self.mass[i],E))
        else:
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*sigma_p)/
                        (2*m_dm*self.mu_p(m_dm)**2))
                rate += (self.frac[i]*prefactor*conversion.conv*
                            vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                            ff.F2(self.A[i],self.mass[i],self.Eprime))
            rate = convolve(self.sigma,E,self.Eprime,rate)  
        return rate


    def dRdE_light(self,m_dm,m_med,sigma_p,E):
        """ Light (vector) mediator model differential event rate.
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        m_med : float
            Mass of mediator particle in MeV.
        sigma_p : float
            DM-nucleon cross section in pb.
        E : array
            Array of energies in keV. 
        """
        rate = 0
        if self.sigma == 0:
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*sigma_p)/
                        (2*m_dm*self.mu_p(m_dm)**2))
                rate += (self.frac[i]*prefactor*conversion.conv*
                            vi.I(vi.v_min(self.mass[i],m_dm,E))*
                            ff.F2(self.A[i],self.mass[i],E)*
                            ((m_med**4)/(2*E*self.mass[i]+m_med**2)**2))
        else:
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*sigma_p)/
                        (2*m_dm*self.mu_p(m_dm)**2))
                rate += (self.frac[i]*prefactor*conversion.conv*
                            vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                            ff.F2(self.A[i],self.mass[i],self.Eprime)*
                            ((m_med**4)/(2*self.Eprime*self.mass[i]+m_med**2)**2))
            rate = convolve(self.sigma,E,self.Eprime,rate)
        return rate

    def dRdE_g_heavy(self, m_dm,g, E):
        """ Heavy (vector) mediator model differential event rate from coupling g.
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        g : float
            Effective coupling constant, in MeV^-2.
        E : array
            Array of energies in keV. 
        """
        rate = 0

        if self.sigma == 0: 
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*9)/
                        (2*np.pi*m_dm))
                coupling = g**2
                rate += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                            vi.I(vi.v_min(self.mass[i],m_dm,E))*
                            ff.F2(self.A[i],self.mass[i],E))
        else: 
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*9)/
                        (2*np.pi*m_dm))
                coupling = g**2
                rate += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                            vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                            ff.F2(self.A[i],self.mass[i],self.Eprime))
            rate = convolve(self.sigma,E,self.Eprime,rate) 

        return rate 

    def dRdE_g_heavy_scalar(self, m_dm, g, theta, E):
        """ Heavy (scalar) mediator model differential event rate from coupling g.
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        g : float
            Effective coupling constant, in MeV^-2.
        theta : float
            Meassure of the ratio of the mediator couplings to neutrons and protons, see 
            See : 'Exploring light mediators with low-threshold direct detection experiments' by Suchita
        E : array
            Array of energies in keV. 
        """
        rate = 0
        if self.sigma == 0:
            for i in range(0,self.nelements):
                prefactor = (((darkmatter.rho_wimp)/(2*np.pi*m_dm))*
                            (self.Z[i]*np.cos(theta)+(self.A[i]-self.Z[i])*np.sin(theta))**2)
                coupling = g**2
                rate += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                            vi.I(vi.v_min(self.mass[i],m_dm,E))*
                            ff.F2(self.A[i],self.mass[i],E))
        else: 
            for i in range(0,self.nelements):
                prefactor = (((darkmatter.rho_wimp)/(2*np.pi*m_dm))*
                            (self.Z[i]*np.cos(theta)+(self.A[i]-self.Z[i])*np.sin(theta))**2)
                coupling = g**2
                rate += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                            vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                            ff.F2(self.A[i],self.mass[i],self.Eprime))
                rate = convolve(self.sigma,E,self.Eprime,rate)          
        return rate
    

    def dRdE_g_light(self,m_dm,m_med,g,E):
        """ Light (vector) mediator model differential event rate from coupling g.
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        m_med : float
            Mass of mediator particle in MeV.
        g : float
            Overall coupling constant, dimensionless.
        E : array
            Array of energies in keV. 
        """
        rate = 0
        if self.sigma == 0:
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*9)/
                     (2*np.pi*m_dm))
                coupling = g**2/(2*self.mass[i]*E+m_med**2)**2
                rate += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                         vi.I(vi.v_min(self.mass[i],m_dm,E))*
                         ff.F2(self.A[i],self.mass[i],E))
        else: 
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*9)/
                     (2*np.pi*m_dm))
                coupling = g**2/(2*self.mass[i]*self.Eprime+m_med**2)**2
                rate += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                         vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                         ff.F2(self.A[i],self.mass[i],self.Eprime))
            rate = convolve(self.sigma,E,self.Eprime,rate) 
        return rate
    

    def dRdE_g_vv(self,m_dm,m_med,g_light,g_heavy,E):
        """ Two (vector) mediator model differential event rate from couplings g_light and g_heavy.
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        m_med : float
            Mass of light mediator particle in MeV.
        g_light : float
            Overall coupling constant for the light mediator, dimensionless.
        g_heavy : float
            Effective coupling constant for the heavy mediator in MeV^-2.
        E : array
            Array of energies in keV. 
        """       
        interference = 0
        #for interference-term:
        if self.sigma == 0:
            rate_light = self.dRdE_g_light(m_dm,m_med,g_light,E)
            rate_heavy = self.dRdE_g_heavy(m_dm,g_heavy,E)
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*9)/
                            (2*np.pi*m_dm))
                coupling = ((2*g_light*g_heavy)/
                            (2*self.mass[i]*E+m_med**2))
                interference += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                                vi.I(vi.v_min(self.mass[i],m_dm,E))*
                                ff.F2(self.A[i],self.mass[i],E))
            rate = rate_heavy+interference+rate_light   
        else:
            rate_light = self.dRdE_g_light(m_dm,m_med,g_light,self.Eprime)
            rate_heavy = self.dRdE_g_heavy(m_dm,g_heavy,self.Eprime)
            for i in range(0,self.nelements):
                prefactor = ((darkmatter.rho_wimp*self.A[i]**2*9)/
                            (2*np.pi*m_dm))
                coupling = ((2*g_light*g_heavy)/
                            (2*self.mass[i]*self.Eprime+m_med**2))
                interference += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                                vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                                ff.F2(self.A[i],self.mass[i],self.Eprime))
            rate = rate_heavy+interference+rate_light 
            rate = convolve(self.sigma,E,self.Eprime,rate) 
        return rate
    
    def dRdE_g_sv(self,m_dm, m_med,g_light,g_heavy,theta,E):
        """ Two (vector) mediator model differential event rate from couplings g_light and g_heavy.
            Attributes
        ----------
        m_dm : float
            Dark matter mass in GeV.
        m_med : float
            Mass of light mediator particle in MeV.
        g_light : float
            Overall coupling constant for the light mediator, dimensionless.
        g_heavy : float
            Effective coupling constant for the heavy mediator in MeV^-2.
        theta : float
            Meassure of the ratio of the mediator couplings to neutrons and protons, see 
            See : 'Exploring light mediators with low-threshold direct detection experiments' by Suchita
        E : array
            Array of energies in keV. 
        """   
        interference = 0
        if self.sigma == 0:
            rate_light = self.dRdE_g_light(m_dm,m_med,g_light,E)
            rate_heavy = self.dRdE_g_heavy_scalar(m_dm,g_heavy,theta,E)
            for i in range(0,self.nelements):
                prefactor = (((darkmatter.rho_wimp*self.A[i]*3)/(2*np.pi*m_dm))
                            *(self.Z[i]*np.cos(theta)+(self.A[i]-self.Z[i])*np.sin(theta)))
                coupling = ((2*g_light*g_heavy)/
                            (2*self.mass[i]*E+m_med**2))
                interference += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                             vi.I(vi.v_min(self.mass[i],m_dm,E))*
                             ff.F2(self.A[i],self.mass[i],E))
            rate = rate_heavy+interference+rate_light
        else:
            rate_light = self.dRdE_g_light(m_dm,m_med,g_light,self.Eprime)
            rate_heavy = self.dRdE_g_heavy_scalar(m_dm,g_heavy,theta,self.Eprime)
            for i in range(0,self.nelements):
                prefactor = (((darkmatter.rho_wimp*self.A[i]*3)/(2*np.pi*m_dm))
                            *(self.Z[i]*np.cos(theta)+(self.A[i]-self.Z[i])*np.sin(theta)))
                coupling = ((2*g_light*g_heavy)/
                            (2*self.mass[i]*self.Eprime+m_med**2))
                interference += (self.frac[i]*prefactor*conversion.conv_g*coupling*
                             vi.I(vi.v_min(self.mass[i],m_dm,self.Eprime))*
                             ff.F2(self.A[i],self.mass[i],self.Eprime))
            rate = rate_heavy+interference+rate_light  
            rate = convolve(self.sigma,E,self.Eprime,rate)
        return rate
### class ends here ---------------------------------------------------------------------------------       

def Ntot(target,rate,args):
    """ Total number of events for a given target.
    ----------
    rate : callable
        Differential recoil spectrum function in the target class
    *args : tuple
        Arguments of rate without the Energy array.
    """ 
    E = np.linspace(target.thr,40,target.Enum)
    r = rate(*args,E)
    N_tot = np.trapz(r,E)*target.exp
    return N_tot

def Ntot_thr(target,rate,thr,args):
    """ Total number of events for a given target and threshold.
    ----------
    rate : callable
        Differential recoil spectrum function in the target class
    thr : array
        Array of thresholds
    *args : tuple
        Arguments of rate without the Energy array.
    """ 
    N_tot = []
    for t in thr:
        E = np.linspace(t,40,target.Enum)
        r = rate(*args,E)
        N_tot.append(np.trapz(r,E)*target.exp)
    return N_tot
        
        




    
    
        
