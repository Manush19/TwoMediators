import numpy as np
import sys,json,os
sys.path.append('..')
import project.tools.statistics as stat
from iminuit import Minuit
import project.quadrantHopping as quadH
from project.likelihoods import LL as ll
import time
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

def reconst_bl(args):
    i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,interference,N = args
    res_mdm = []
    res_mzl = []
    res_bl = []
    res_ll = []
    mdm = Mdm[i]
    bl_guess = 1.
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = barys[i,j]
        llike = ll(interference,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl)
        m = Minuit(llike.nll_blglgh, bl = bl_guess, nlgl = -np.log10(actual['gl']), 
                   nlgh = -np.log10(np.abs(actual['gh'])) )
        m.fixed['nlgl'] = True
        m.fixed['nlgh'] = True
        m.limits['bl'] = (.3,3.)
        m.errordef = Minuit.LIKELIHOOD
        m.print_level = 0
        min_res,m = quadH.quadhop(m, ['bl'], [bl_guess])
        res_mdm.append(mdm)
        res_mzl.append(mzl)
        res_ll.append(min_res['globalfun'])
        res_bl.append(min_res['globalmin'][0])
        bl_guess = min_res['globalmin']
    Res = [i,res_mdm,res_mzl,res_ll,res_bl]
    return Res

def reconst_blgl(args):
    i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,interference,N = args
    res_mdm = []
    res_mzl = []
    res_bl = []
    res_gl = []
    res_ll = []
    mdm = Mdm[i]
    bl_guess = 1.
    nlgl_guess = 10.
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = barys[i,j]
        llike = ll(interference,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl)
        m = Minuit(llike.nll_blglgh, bl = bl_guess, nlgl = nlgl_guess, 
                   nlgh = -np.log10(np.abs(actual['gh'])))
        m.fixed['nlgh'] = True
        m.limits['bl'] = (0.3,3.)
        m.limits['nlgl'] = (5.,15.)
        m.errordef = Minuit.LIKELIHOOD
        m.print_level = 0
        min_res,m = quadH.quadhop(m, ['bl','nlgl'], [bl_guess, nlgl_guess])
        res_mdm.append(mdm)
        res_mzl.append(mzl)
        res_ll.append(min_res['globalfun'])
        res_bl.append(min_res['globalmin'][0])
        res_gl.append(10**-min_res['globalmin'][1])
        bl_guess,nlgl_guess = min_res['globalmin']
    Res = [i,res_mdm,res_mzl,res_ll,res_bl,res_gl]
    return Res

def reconst_blglgh(args):
    i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,interference,N = args
    res_mdm = []
    res_mzl = []
    res_bl = []
    res_gl = []
    res_gh = []
    res_ll = []
    mdm = Mdm[i]
    bl_guess = 1.
    nlgl_guess = 10.
    nlgh_guess = 12.
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = barys[i,j]
        llike = ll(interference,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl)
        m = Minuit(llike.nll_blglgh, bl = bl_guess, nlgl = nlgl_guess, nlgh = nlgh_guess)
        m.limits['bl'] = (0.3,3.)
        m.limits['nlgl'] = (5.,15.)
        m.limits['nlgh'] = (5.,15.)
        m.errordef = Minuit.LIKELIHOOD
        min_res,m = quadH.quadhop(m, ['bl','nlgl','nlgh'],[bl_guess,nlgl_guess,nlgh_guess])
        res_mdm.append(mdm)
        res_mzl.append(mzl)
        res_ll.append(min_res['globalfun'])
        res_bl.append(min_res['globalmin'][0])
        res_gl.append(10**-min_res['globalmin'][1])
        res_gh.append(llike.d_sign*10**-min_res['globalmin'][2])
        bl_guess,nlgl_guess,nlgh_guess = min_res['globalmin']
    Res = [i,res_mdm,res_mzl,res_ll,res_bl,res_gl,res_gh]
    return Res

def reconst_blrh(args):
    i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,interference,N = args
    res_mdm = []
    res_mzl = []
    res_bl = []
    res_rh = []
    res_ll = []
    mdm = Mdm[i]
    bl_guess = 1.
    rh_guess = .3
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = barys[i,j]
        llike = ll(interference,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl)
        m = Minuit(llike.nll_blglghrh, bl = bl_guess, nlgl = -np.log10(actual['gl']),
                   nlgh = -np.log10(np.abs(actual['gh'])), rh = rh_guess)
        m.fixed['nlgl'] = True
        m.fixed['nlgh'] = True
        m.limits['bl'] = (0.3,3.)
        m.limits['rh'] = (.2,.6)
        m.errordef = Minuit.LIKELIHOOD
        min_res,m = quadH.quadhop(m, ['bl','rh'],[bl_guess,rh_guess])
        res_mdm.append(mdm)
        res_mzl.append(mzl)
        res_ll.append(min_res['globalfun'])
        res_bl.append(min_res['globalmin'][0])
        res_rh.append(min_res['globalmin'][1])
        bl_guess,rh_guess = min_res['globalmin']
    Res = [i,res_mdm,res_mzl,res_ll,res_bl,res_rh]
    return Res

def reconst_blglrh(args):
    i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,interfernce,N = args
    res_mdm = []
    res_mzl = []
    res_bl = []
    res_gl = []
    res_rh = []
    res_ll = []
    mdm = Mdm[i]
    bl_guess = 1.
    nlgl_guess = 10.
    rh_guess = .3
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = barys[i,j]
        llike = ll(interfernce,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl)
        m = Minuit(llike.nll_blglghrh, bl = bl_guess, nlgl = nlgl_guess,
                   nlgh = -np.log10(np.abs(actual['gh'])), rh = rh_guess)
        m.fixed['nlgh'] = True
        m.limits['bl'] = (0.3,3.)
        m.limits['nlgl'] = (5.,15.)
        m.limits['rh'] = (.2,.6)
        m.errordef = Minuit.LIKELIHOOD
        min_res,m = quadH.quadhop(m, ['bl','nlgl','rh'],[bl_guess,nlgl_guess,rh_guess])
        res_mdm.append(mdm)
        res_mzl.append(mzl)
        res_ll.append(min_res['globalfun'])
        res_bl.append(min_res['globalmin'][0])
        res_gl.append(10**-min_res['globalmin'][1])
        res_rh.append(min_res['globalmin'][2])
        bl_guess,nlgl_guess,rh_guess = min_res['globalmin']
    Res = [i,res_mdm,res_mzl,res_ll,res_bl,res_gl,res_rh]
    return Res

def reconst_blglghrh(args):
    i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,interfernce,N = args
    res_mdm = []
    res_mzl = []
    res_bl = []
    res_gl = []
    res_gh = []
    res_rh = []
    res_ll = []
    mdm = Mdm[i]
    bl_guess = 1.
    nlgl_guess = 10.
    nlgh_guess = 12.
    rh_guess = .3
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = barys[i,j]
        llike = ll(interfernce,target,binnedsample,ni,bbg,b0,b1,b2,mdm,mzl)
        m = Minuit(llike.nll_blglghrh, bl = bl_guess, nlgl = nlgl_guess,
                   nlgh = nlgh_guess, rh = rh_guess)
        m.limits['bl'] = (0.3,3.)
        m.limits['nlgl'] = (5.,15.)
        m.limits['nlgh'] = (5.,15.)
        m.limits['rh'] = (.2,.6)
        m.errordef = Minuit.LIKELIHOOD
        min_res,m = quadH.quadhop(m, ['bl','nlgl','nlgh','rh'],[bl_guess,nlgl_guess,nlgh_guess,rh_guess])
        res_mdm.append(mdm)
        res_mzl.append(mzl)
        res_ll.append(min_res['globalfun'])
        res_bl.append(min_res['globalmin'][0])
        res_gl.append(10**-min_res['globalmin'][1])
        res_gh.append(llike.d_sign*10**-min_res['globalmin'][2])
        res_rh.append(min_res['globalmin'][3])
        bl_guess,nlgl_guess,nlgh_guess,rh_guess = min_res['globalmin']
    Res = [i,res_mdm,res_mzl,res_ll,res_bl,res_gl,res_gh,res_rh]
    return Res

def min_ll(samplefile,baryfile,Nbins = 30, fin = 5, N = 40,interference = 'd',
           reconst_key = 'blglgh',id = ''):
    """Program for minimizing extended binned likelihood at all Mdm-Mzl gridpoints.
    ----------------
    samplefile: file name of the mock sample (json file)
    baryfile: file name where barys are stored (json file)
    Nbins: Number of bins for binning the sample. Should be same as the Nbins used for 
           calculating barys
    fin: Accuracy/ number of energy points to be considered withing each bin. Should be
         same as the fin used for calculating barys.
    N: Number of grid points (NxN grid of Mdm-Mzl). Should be the same as used
       used for calculating barys.
    interference: 'c' or 'd' for constructive or destructive interference to be considered
                  for reconstruction.
    reconst_key: key that determines the nuisance parameters and the fixed parameters in
                 reconstruction. 
                 'bl' ------> nuisance params: bl, fixed params: gl,gh
                 'blgl' ----> nuisance params: bl,gl, fixed params: gh
                 'blglgh'---> nusiance params: bl,gl,gh, fixed params: 
                 'blrh' ----> nuisance params: bl,rh fixed params: gl,gh
                 'blglrh' --> nuisance params: bl,gl,rh fixed params: gh
                 'blglghrh'-> nuisance params: bl,gl,gh,rh fixed params:
    id: str, extra identification for the file name to store the results.
    """
    starttime = time.time()
    whereto = os.path.join(os.getcwd(),'../Output/')
    
    inputdata = json.load(open(whereto+samplefile,'r'))
    targetdata = inputdata['target']
    material = targetdata['material']
    exp = targetdata['exposure']
    thr = targetdata['thr']
    res = targetdata['sigma']
    roi = targetdata['roi']
    bg = targetdata['bg']
    
    vdf = np.array(inputdata['vdf'])
    ve = np.array(inputdata['ve'])
    vesc = inputdata['vesc']
    vcir = inputdata['vcir']
    I_index = inputdata['I_index']
    target = stat.target(*material, exposure=exp, thr=thr, sigma=res, bg=bg, roi=roi)
    if vdf[0] == 'SHM':
        target.initialize_vdf(I_index = I_index, vesc = vesc, vcir = vcir)
    else:
        target.initialize_vdf(I_index = 2, VDF = vdf, V = ve, vesc = vesc, vcir = vcir)
        
    modeldata = inputdata['model']
    actual = {'mdm':modeldata['mdm'],
              'mzl':modeldata['mZl'],
              'gl':modeldata['gl'],
              'gh':modeldata['gheff'],
              'bl':bg}
        
    sample = np.array(inputdata['sample'])
    
    bsample = stat.binned.binsample(sample,Nbins,fin)
    bbg = stat.binned.bintbg(target,bsample)
    binnedsample = bsample['binnedsample']
    ni = binnedsample[0]
    
    Mdm = np.linspace(1.,10.,N)
    Mzl = np.linspace(1.,20.,N)
    
    barys = np.array(json.load(open(whereto+baryfile,'r')))
    
    if barys.shape[0] != N:
        print ('N = %i and shape(barys) = %s do not match'%(N.barys.shape))
        return None
    
    if reconst_key == 'bl':
        func = reconst_bl
    elif reconst_key == 'blgl':
        func = reconst_blgl
    elif reconst_key == 'blglgh':
        func = reconst_blglgh
    elif reconst_key == 'blrh':
        func = reconst_blrh
    elif reconst_key == 'blglrh':
        func = reconst_blglrh
    elif reconst_key == 'blglghrh':
        func = reconst_blglghrh
    
    pool = Pool()
    results = pool.map(func,[[i,barys,target,binnedsample,ni,bbg,Mdm,Mzl,actual,
                       interference,N] for i in range(N)])
    
    reconst_file_name = whereto + samplefile[:-5]+'_reconst_%s.json'%id
    json.dump(results, open(reconst_file_name,'w'),indent = 2)
    
    print (f'Total work completed in {(time.time()-starttime)/60.} mins')
    
    
    
    
    
    
    
    