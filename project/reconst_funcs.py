import numpy as np
import matplotlib.pyplot as plt
import sys,json
import time
from iminuit import Minuit
sys.path.append('..')
import project.tools.statistics as stat1
import project.QuadrantHopping as quadH
from project.Likelihoods import LL as ll
from project.plot_ll import plot_all as pall

def b_arys(bsample,target,mdm,mzl):
    b0 = stat1.binned.bint0(target,bsample,mdm)
    b1 = stat1.binned.bint1(target,bsample,mdm,mzl)
    b2 = stat1.binned.bint2(target,bsample,mdm,mzl)
    return b0,b1,b2
def e_arys(bsample,Nbins):
	binnedE = bsample['binnedE']
	accuracy = bsample['accuracy']
	eary = []
	for i in range(Nbins):
		eary.append(binnedE[int(i*accuracy)])
	Eary = []
	for i in range(len(eary)-1):
		Eary.append(0.5*(eary[i]+eary[i+1]))
	diff = Eary[-1]-Eary[-2]
	Eary.append(Eary[-1]+diff)
	return np.array(Eary)

def reconstruct_gl(local_Mdm,Mzl,nlgl_guess0,b0,b1,b2,stt,bsample,target,bbg,actual):
	N  = len(Mzl)
	gl_dm = np.zeros([local_Mdm.shape[0],N])
	gl_zl = np.zeros([local_Mdm.shape[0],N])
	gl_ll = np.zeros([local_Mdm.shape[0],N])
	gl_gl = np.zeros([local_Mdm.shape[0],N])
	bl = actual['bl']
	nlgh = -np.log10(np.abs(actual['gh']))
	for i in range(len(local_Mdm)):
		mdm = local_Mdm[i]
		for j in range(N):
			mzl = Mzl[j]
			
			try:
				nlgl_guess = nlgl_guess
			except:
				nlgl_guess = nlgl_guess0
			
			b0gl = b0[i,j,:,:]
			b1gl = b1[i,j,:,:]
			b2gl = b2[i,j,:,:]
			
			llike = ll('d', bsample, target, b0gl,b1gl,b2gl, mdm, mzl, bbg)
			
			m = Minuit(llike.nll_blglgh, bl=bl, nlgl=nlgl_guess,nlgh = nlgh)
			m.fixed['bl','nlgh'] = True
			m.limits['nlgl'] = (5.,15.)
			m.errordef = Minuit.LIKELIHOOD
			min_res,m = quadH.quadhop(m, ['nlgl'],[nlgl_guess])
			
			gl_dm[i,j] = mdm
			gl_zl[i,j] = mzl
			gl_ll[i,j] = min_res['globalfun']
			gl_gl[i,j] = min_res['globalmin'][0]
			
			print (f'time taken = {round((time.time() - stt)/60,1)} min and {round((time.time() - stt)%60,1)} sec')
		
	return [gl_dm,gl_zl,gl_ll,gl_gl]
	
	
	
