#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import project.tools.statistics as stat1


# In[8]:

class LL:
    def __init__(self,interference,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl,**kwargs):
        """
        If vcir is a free parameter, each of b0,b1,b2 should be a list which contain N number of 
        b0's,b1's and b2's [numpy arrays] corresponding to a given mdm and mzl and N values of vcir.
        Note: b0(mdm,mzl) = [b0(mdm,mzl,vcir_0), b0(mdm,mzl,vcir_1), ..., b0(mdm,mzl,vcir_n)], if vcir is
        a free parameter. else, b0 = b0(mdm,mzl)
        **Important**: b0's corresponding to vcir's should be given as a list and not as numpy arrays.
        
        kwargs:
        Vcir: if b0 is a list. i.e., vcir is a free parameter, then the lsit of vcir's to which 
        the b0's correspond to should be provided. 
        """
        self.target = target
        self.binnedsample = binnedsample
        self.ni = ni
        self.interference = interference
        self.mdm = mdm
        self.mzl = mzl
        self.bbg = bbg
        if self.interference == 'd':
            self.d_sign = -1.
        else:
            self.d_sign = 1.
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        if isinstance(b0, list):
            if not isinstance(b1, list) or not isinstance(b2, list):
                print ('b0 is list, but b1 or  b2 is not')
            if not len(b1) == len(b0) or not len(b2) == len(b0):
                print ('the sizes of lists of b0, b1, b2 do not match')
            if not 'Vcir' in kwargs:
                print ('b0 is a list so Vcir should be provided!')
            else:
                self.Vcir = np.array(kwargs.get('Vcir'))
        
    def nll_blglgh(self,bl,nlgl,nlgh):
        gl = 10**-nlgl
        gh = self.d_sign*10**-nlgh
        lambd_ary = stat1.binned.bintot(self.target,self.mdm,self.mzl,gl,gh,
                                        self.b0,self.b1,self.b2,self.bbg,bl)
        ll = -np.sum(lambd_ary)
        ll += (np.log(lambd_ary)*self.ni).sum()
        return -ll

    def nll_blglghrh(self,bl,nlgl,nlgh,rh):
        gl = 10**-nlgl
        gh = self.d_sign*10**-nlgh
        lambd_ary = stat1.binned.bintotrho(self.target,self.mdm,self.mzl,gl,gh,
                                          self.b0,self.b1,self.b2,self.bbg,bl,rh)
        ll = -np.sum(lambd_ary)
        ll += (np.log(lambd_ary)*self.ni).sum()
        return -ll
    
    def nll_blglghvc(self,bl,nlgl,nlgh,vcir):
        idx = np.argmin(self.Vcir - vcir)
        b0,b1,b2 = b0[idx],self.b1[idx],self.b2[idx]            
        gl = 10**-nlgl
        gh = self.d_sign*10**-nlgh
        lambd_ary = stat1.binned.bintot(self.target,self.mdm,self.mzl,gl,gh,
                                        b0,b1,b2,self.bbg,bl)
        ll = -np.sum(lambd_ary)
        ll += (np.log(lambd_ary)*self.ni).sum()
        return -ll
        
            
class LL_old:
	def __init__(self,interference, bsample, target, b0,b1,b2,mdm,mzl,bbg):
		self.interference = interference
		self.bsample = bsample
		self.target = target
		self.bbg = bbg
		self.b0 = b0
		self.b1 = b1
		self.b2 = b2
		self.mdm = mdm
		self.mzl = mzl
		if self.interference == 'd':
			self.d_sign = -1.
		else:
			self.d_sign = 1.
		self.binnedsample = self.bsample['binnedsample']
		self.Nbins = self.bsample['binnum']
	def nll_blglgh(self,bl,nlgl,nlgh):
		gl = 10**-nlgl
		gh = self.d_sign*10**-nlgh
		lambd_ary = stat1.binned.bintot(self.target,self.mdm,self.mzl,gl,gh,self.b0,self.b1,self.b2,self.bbg,bl)
		ll = -np.sum(lambd_ary)
		for i in range(self.Nbins):
			lambd_i = lambd_ary[i]
			if np.isfinite(lambd_i) == False:
				ll = -np.inf
				break
			n_i = self.binnedsample[0][i]
			ll += n_i*np.log(lambd_i)
			if np.isfinite(ll) == False:
				ll = -np.inf
				break
		return -ll