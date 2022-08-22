# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np
import scipy as sc
import scipy.special as sp
from project.constants import cosmo 
import project.tools.statistics as stat1

z = np.sqrt((3*cosmo.v_esc**2)/(2*cosmo.w**2))
N = sp.erf(z)-2/(np.sqrt(np.pi))*z*np.exp(-z**2)
eta = np.sqrt((3*cosmo.v_earth**2)/(2*cosmo.w**2))
fac = 1/(N*eta)*(3/(2*np.pi*cosmo.w**2))**0.5


def req_consts(vesc = cosmo.v_esc, w = cosmo.w, vearth = cosmo.v_earth):
    z = np.sqrt((3*vesc**2)/(2*w**2))
    N = sp.erf(z)-2/(np.sqrt(np.pi))*z*np.exp(-z**2)
    eta = np.sqrt((3*vearth**2)/(2*w**2))
    fac = 1/(N*eta)*(3/(2*np.pi*w**2))**0.5
    return (z,N,eta,fac)

def mu_N(m_N,m_dm):
    return (m_dm*m_N)/(m_dm+m_N)

def x_min(vmin):
    return np.sqrt((3*vmin**2)/(2*cosmo.w**2))

def v_min(m_N,m_dm,E):
    return np.sqrt((E*1e-6*m_N)/(2*mu_N(m_N,m_dm)**2))*cosmo.c


def I(vmin):
    #print ('Its in 0')
    integral = np.zeros(np.size(vmin))
    for j in range(0,np.size(vmin)):
        if x_min(vmin[j])<(z-eta):
            integral[j] = fac*(0.5*np.sqrt(np.pi)*(sp.erf(x_min(vmin[j])+eta)-sp.erf(x_min(vmin[j])-eta))-2*eta*np.exp(-z*z))
        elif x_min(vmin[j])<(z+eta):
            integral[j] = fac*(0.5*np.sqrt(np.pi)*(sp.erf(z)-sp.erf(x_min(vmin[j])-eta))-np.exp(-z**2)*(z+eta-x_min(vmin[j])))
        else:
            integral[j] = 0
    return integral

# def AM(vmin):
    

def I1(vmin,vesc = cosmo.v_esc, vrms = cosmo.w, vearth = cosmo.v_earth):
    #print ('Its in 1')
    z,N,eta,fac = req_consts(vesc, vrms, vearth)
    integral = np.zeros(np.size(vmin))
    for j in range(0,np.size(vmin)):
        if x_min(vmin[j])<(z-eta):
            integral[j] = fac*(0.5*np.sqrt(np.pi)*(sp.erf(x_min(vmin[j])+eta)-sp.erf(x_min(vmin[j])-eta))-2*eta*np.exp(-z*z))
        elif x_min(vmin[j])<(z+eta):
            integral[j] = fac*(0.5*np.sqrt(np.pi)*(sp.erf(z)-sp.erf(x_min(vmin[j])-eta))-np.exp(-z**2)*(z+eta-x_min(vmin[j])))
        else:
            integral[j] = 0
    return integral
    
    
#--------------------------------  I2 -------------------------------------------------------
def Cmax(v,vesc,vearth):
    return (vesc**2 - v**2 - vearth**2)/(2*v*vearth)
	
def v_plus(vmin,vesc,vearth):
    return np.min([vmin,vesc+vearth])

def v_minus(vmin,vesc,vearth):
    return np.min([vmin,vesc-vearth])	
	
def F1D(v, VDF ,vesc, vearth):
    """
    One dimensional speed distribution in Earth's reference frame, using v and VDF in earth's 
    reference frame.
    TO DO: include theta integration. the addition in exponential power is vector addition!!!
    """
    cmax = Cmax(v,vesc,vearth)
    t1 = np.heaviside(vesc + vearth - v, 0)*2.*np.pi*v**2 * (cmax + 1)
    t2 = np.heaviside(vesc - vearth - v, 0)*2.*np.pi*v**2 * (cmax - 1)
    F1D_fact = (t1 - t2)
    return F1D_fact*VDF
	
def I2(vE, VDF_E, Vmin, vesc, vearth):
    """
    v is the DM velocity in earth's reference frame.
    VDF: is the 1 dimensional DM speed distribution in earth's reference frame.
    Vmin is the mininum velocity depended on E_r
    vesc is the escape velocity in galactic reference frame.
    vearth is the vel of earth in galactic reference frame.
    """
    #v = v - vearth # converted to earth refernce frame.
    integral = np.zeros(np.size(Vmin))
    for j in range(0,np.size(Vmin)):
        vmin = Vmin[j]
        #f1d = F1D(vE, VDF_E, vmin, vesc, vearth)
		
        low1 = v_plus(vmin,vesc,vearth)
        hig1 = vesc + vearth
        if low1 <= hig1:
            idx1 = np.where((vE >= low1) & (vE <= hig1))
            idg1 = VDF_E[idx1]
            v1 = vE[idx1]
            int1 = np.trapz(idg1/v1, v1)
        else:
            int1 = 0
		
        low2 = v_minus(vmin,vesc,vearth)
        hig2 = vesc - vearth
        if low2 >= hig2:
            idx2 = np.where((vE >= low2) & (vE <= hig2))
            idg2 = VDF_E[idx2]
            v2 = vE[idx2]
            int2 = np.trapz(idg2/v2, v2)
        else:
            int2 = 0
		
        integral[j] = (int1 - int2)
        
        
    return integral


class VDF:
    def __init__(self,**kwargs):
        """
        Throughout vcir represents the asymptotic value of the circular rotation speed.
        since it is generally assumed that the halo reached its flat part at sun's location,
        it is often (and here) assumed that vsun (circular speed at sun's location) = vcir
        """
        self.vcir0 =  220.0
        self.vesc0 = 544.0
        self.vrms0 = np.sqrt(3./2.)*self.vcir0
        self.vert0 = self.vcir0*(1.05 + 0.07)
        if 'vgal' in kwargs:
            self.vgal = kwargs.get('vgal')
        else:
            self.vgal = np.linspace(0.1, 1000., 1000)
        if 've' in kwargs:
            self.ve = kwargs.get('ve')
        else:
            self.ve = np.linspace(0.1, 1000., 1000)
        if 'vmin' in kwargs:
            self.vmin = kwargs.get('vmin')
        else:
            self.vmin = np.linspace(0.1, 1000., 1000)
        
    def Vrms(self,vcir):
        """
        dispersion velocity
        """
        return np.sqrt(3./2.)*vcir
           
    def Vert(self,vcir):
        """
        velocity of earth in gal ref frame
        """
        return vcir*(1.05 + 0.07)
        
    def unpack_vs(self,kwargs):
        if 'vesc' in kwargs:
            vesc = kwargs.get('vesc')
        else:
            vesc = self.vesc0
        if 'vcir' in kwargs:
            vcir = kwargs.get('vcir')
        else:
            vcir = self.vcir0
        return vesc,vcir
    
    def mod_vgal(self,vdf):
        vdf = vdf*self.vgal**2
        return vdf/np.trapz(vdf,np.abs(self.vgal))
    
    def shm_gal(self,mod = False,**kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vgal = self.vgal
        Nesc = sp.erf(vesc/vcir) - 2*vesc*np.exp(-(vesc/vcir)**2)/np.sqrt(np.pi)/vcir
        shm_gf = np.heaviside(vesc - vgal, 0)*np.exp(-(vgal/vcir)**2)/(vcir*np.sqrt(np.pi))**3/Nesc
        if not mod:
            return shm_gf/np.trapz(shm_gf,self.vgal)
        else:
            return self.mod_vgal(shm_gf)
    
    def shm_ert(self,**kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vert = self.Vert(vcir)
        ve = self.ve
        z = ve/vcir
        zert = vert/vcir
        zesc = vesc/vcir
        Nesc = sp.erf(vesc/vcir) - (2.*vesc)/(3.*np.sqrt(np.pi)*vcir)*3.*np.exp(-vesc**2/vcir**2)
        fact = 1./(np.sqrt(np.pi)*Nesc*vcir*zert)
        t1 = np.heaviside(zesc + zert - z,0)*z*(np.exp(-(z-zert)**2) - np.exp(-zesc**2))
        t2 = np.heaviside(zesc - zert - z,0)*z*(np.exp(-(z+zert)**2) - np.exp(-zesc**2))
        shm_ef = fact*(t1 - t2)
        return shm_ef#/np.trapz(shm_ef,ve)
    
    def vdf_gal(self, V_gal, VDF_gal, newvgal = 'none',mod = False):
        """
        vgal is the velocity in galactic reference frame where the vdfs in galactic reference 
        are required.
        V_gal and VDF_gal are the data of vdf in galactic reference frame which is used to 
        interpolate and obtain vdfs are vgal.
        """
        if isinstance(newvgal, str):
            if newvgal == 'none':
                newvgal = self.vgal
        newfunc = sc.interpolate.interp1d(V_gal, VDF_gal, fill_value = 0, bounds_error = False)
        if not mod:
            return newfunc(newvgal)
        else:
            return self.mod_vgal(newfucn(newvgal))
        
    def vdf_1D_ert(self,fgal,**kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vert = self.Vert(vcir)
        vgal = self.vgal
        F_E = []
        for v in self.ve:
            cmax = (vesc**2 - v**2 - vert**2)/(2.*v*vert)
            
            x_plus = np.linspace(-1, cmax, 1000)
            newv_plus = np.sqrt(v**2 + vert**2 + 2*v*vert*x_plus)
            f_xplus = self.vdf_gal(self.vgal, fgal, newvgal = newv_plus)
            int_plus = np.trapz(f_xplus, x_plus)
            t_plus = np.heaviside(vesc + vert - v, 0)*int_plus

            x_minus = np.linspace(1, cmax, 1000)
            newv_minus = np.sqrt(v**2 + vert**2 + 2.*v*vert*x_minus)
            f_xminus = self.vdf_gal(self.vgal, fgal, newvgal = newv_minus)
            int_minus = np.trapz(f_xminus, x_minus)
            t_minus = np.heaviside(vesc - vert - v,0)*int_minus
            
            F_E.append(v**2 * (t_plus - t_minus)*2*np.pi)
            
        F_E = np.array(F_E)
        return F_E/np.trapz(F_E)
    
    def eta(self,vdf,*args):
        ve = self.ve
        vmin = self.vmin
        vesc,vcir = self.unpack_vs(args)
        vert = self.Vert(vcir)
        eta = []
        for v in vmin:
            v_plus = np.min([v, vesc + vert])
            v_minus = np.min([v, vesc - vert])
            
            low1 = v_plus
            hig1 = vesc + vert
            if low1 <= hig1:
                idx1 = np.where((ve >= low1) & (ve <= hig1))
                idg1 = vdf[idx1]
                v1 = ve[idx1]
                int1 = np.trapz(idg1/v1, v1)
            else:
                int1 = 0
            low2 = v_minus
            hig2 = vesc - vert
            if low2 >= hig2:
                idx2 = np.where((ve >= low2) & (ve <= hig2))
                idg1 = vdf[idx2]
                v2 = ve[idx2]
                int2 = np.trapz(idg1/v2, v2)
            else:
                int2 = 0
            eta.append(int1 - int2)
        return np.array(eta)
    
    def initialize_target(self,material,exp,thr,res,bg,roi):
        self.target = stat1.target(*material, exposure = exp, thr = thr, 
                                  sigma = res, bg = bg, roi = roi)
        
    def initialize_index(self,I_index,vdf,**kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        self.target.initialize_vdf(I_index = I_index, VDF = vdf, V = self.ve, vesc = vesc, vcir = vcir)
        
    def diffrecoil(self,mdm,mzl,gl,gh,E):
        return self.target.diffrecoil(mdm,mzl,gl,gh,E)
    
    def Vdf_ani_gal(self, vr, vth, vphi, vcir, beta, mod = False):
        sig_r = 3*vcir**2/(2*(3-2*beta))
        sig_th = 3*(1-beta)*vcir**2/(2*(3-2*beta))
        sig_phi = sig_th
        fgal = np.exp(-(vr**2/(2*sig_r))-(vth**2/(2*sig_th))-(vphi**2/(2*sig_phi)))
        if not mod:
            return fgal
        else:
            return self.mod_vgal(fgal)
    
    def Vdf_ano_1D_ert(self,beta,**kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vert = self.Vert(vcir)
        vr = lambda v,th,phi: v*np.sin(th)*np.sin(phi)
        vth = lambda v,th,phi: v*np.sin(th)*np.cos(phi)
        vphi = lambda v,th: v*np.cos(th)
        phi = np.linspace(0,2.*np.pi,100)
        FE = []
        for v in self.ve:
            cmax = (vesc**2 - v**2 - vert**2)/(2.*v*vert)
            if cmax > 1: cmax = 1
            if cmax < -1: cmax = -1
                
            if cmax == -1:
                t_plus = 0
            else:
                costh_plus = np.linspace(-1,cmax,100)
                f_costh_plus = []
                for cth_plus in costh_plus:
                    th_plus = np.arccos(cth_plus)
                    vdf_ani_gal_plus = self.Vdf_ani_gal(vr(v,th_plus,phi),
                                                       vth(v,th_plus,phi),
                                                       vphi(v,th_plus)+vert,
                                                       vcir, beta)
                    f_costh_plus.append(np.trapz(vdf_ani_gal_plus,phi))
                f_costh_plus = np.array(f_costh_plus)
                int_plus = np.trapz(f_costh_plus, costh_plus)
                t_plus = np.heaviside(vesc + vert - v, 0)*int_plus
                
            if cmax == 1:
                t_mins = 0
            else:
                costh_mins = np.linspace(1, cmax, 100)
                f_costh_mins = []
                for cth_mins in costh_mins:
                    th_mins = np.arccos(cth_mins)
                    vdf_ani_gal_mins = self.Vdf_ani_gal(vr(v,th_mins,phi),
                                                  vth(v,th_mins,phi),
                                                  vphi(v,th_mins)+vert,
                                                  vcir, beta)
                    f_costh_mins.append(np.trapz(vdf_ani_gal_mins,phi))
                f_costh_mins = np.array(f_costh_mins)
                int_mins = np.trapz(f_costh_mins, costh_mins)
                t_mins = np.heaviside(vesc - vert - v, 0)*int_mins
                
            FE.append(v**2 * (t_plus - t_mins))
        FE = np.array(FE)
        return FE/np.trapz(FE,self.ve)
    
    def mao_2013_gal(self,p,mod = False,**kwargs):
        """
        Based on Mao et al. 2013 (ApJ 764 35).
        For vcir = 220., vesc = 544., their fit in Figure 4 shows that
        1 <= p <= 3.5
        """
        vesc,vcir = self.unpack_vs(kwargs)
        vdf = np.zeros(self.vgal.shape)
        idx = np.where(self.vgal <= vesc)
        vdf[idx] = np.exp(-np.abs(self.vgal[idx])/vcir)*(vesc**2 - np.abs(self.vgal[idx])**2)**p
        vdf = vdf/np.trapz(vdf,self.vgal)
        if not mod:
            return vdf
        else:
            return self.mod_vgal(vdf)
        
    def mao_2013_ert(self, p, **kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vdf_gf = self.mao_2013_gal(p,**kwargs)
        return self.vdf_1D_ert(vdf_gf,**kwargs)
    
    def lisanti_2011_gal(self, k, mod = False, **kwargs):
        """
        Based on Lisanti et al 2011, PRD 023519:
        The isotropic velocity distribution function corresponding to 
        double power law density profiles. This distibution is an ansatz 
        is that it satisfies the Jeans theorem for an equilibrated system
        and goes to zero at the escape velocity
        For outer dark matter profile with gamma nearly [3,5] 
        (note: for NFW gamma = 3), k = [1.5,3.5].
        """
        vesc,vcir = self.unpack_vs(kwargs)
        vdf = np.zeros(self.vgal.shape)
        idx = np.where(self.vgal <= vesc)
        vdf[idx] = (np.exp((vesc**2 - self.vgal[idx]**2)/(k*vcir**2))-1)**k
        vdf = vdf/np.trapz(vdf,self.vgal)
        if not mod:
            return vdf
        else:
            return self.mod_vgal(vdf)
        
    def lisanti_2011_ert(self, k, **kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vdf_gf = self.lisanti_2011_gal(k,**kwargs)
        return self.vdf_1D_ert(vdf_gf,**kwargs)
    
    def tsallis_gal(self, q, mod = False, **kwargs):
        """
        This is from eq 2.3 in Ling et al 2010 paper. 
        1/3 < q < 5/3. Lower bound from Ling 2000 paper and
        upper bound from Hansen 2006 paper.
        """
        vesc,vcir = self.unpack_vs(kwargs)
        if q != 1:
            vdf = np.zeros(self.vgal.shape)
            if q < 1:
                v_tsa_esc = vcir/np.sqrt(1-q)
            else:
                v_tsa_esc = self.vgal[-1]
            idx = np.where(self.vgal < v_tsa_esc)
            vdf[idx] = (1-(1-q)*(self.vgal[idx]**2/vcir**2))**(q/(1-q))
        else:
            vdf = self.shm_gal(kwargs)
        vdf = vdf/np.trapz(vdf,self.vgal)
        if not mod:
            return vdf
        else:
            return self.mod_vgal(vdf)
        
    def tsallis_ert(self, q, **kwargs):
        vesc,vcir = self.unpack_vs(kwargs)
        vdf_gf = self.tsallis_gal(q,**kwargs)
        return self.vdf_1D_ert(vdf_gf,**kwargs)
    
        
    
    