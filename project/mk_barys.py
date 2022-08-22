"""
from multiprocessing import Pool
import time
import numpy as np

y = np.ones([2,2])

def f(x):
	return y + x
	
	
p = Pool()
result = p.map(f,np.linspace(1,10,10))
p.close()

res = np.array(result)

print (res.shape)
"""

import numpy as np
import sys,json,os
sys.path.append('../')
import project.tools.statistics as stat
from multiprocessing import Pool
import time

def b_arys(bsample,target,mdm,mzl):
    b0 = stat.binned.bint0(target,bsample,mdm)
    b1 = stat.binned.bint1(target,bsample,mdm,mzl)
    b2 = stat.binned.bint2(target,bsample,mdm,mzl)
    return b0,b1,b2

def mkbary(args):
    indx,N,Mdm,Mzl,bsample,target = args
    mdm = Mdm[indx]
    res = []
    for j in range(N):
        mzl = Mzl[j]
        b0,b1,b2 = b_arys(bsample,target,mdm,mzl)
        res.append([b0.tolist(),b1.tolist(),b2.tolist()])
    return res

def mk_barys(samplefile, Nbins = 30, fin = 5, N = 40, id = ''):
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
    target = stat.target(*material, exposure = exp, thr = thr, sigma = res, bg = bg,
                          roi = roi)
    
    vdf = np.array(inputdata['vdf'])
    ve = np.array(inputdata['ve'])
    vesc = inputdata['vesc']
    vcir = inputdata['vcir']
    I_index = inputdata['I_index']
    if vdf[0] == 'SHM':
        target.initialize_vdf(I_index = I_index, vesc = vesc, vcir = vcir)
    else:
        target.initialize_vdf(I_index = 2, VDF = vdf, V = ve, vesc = vesc, vcir = vcir)
        
    sample = np.array(inputdata['sample'])
    
    bsample = stat.binned.binsample(sample,Nbins,fin)
    bbg = stat.binned.bintbg(target,bsample)
    
    Mdm = np.linspace(1.,10.,N)
    Mzl = np.linspace(1.,20.,N)
    
    pool = Pool()
    results = pool.map(mkbary,[[i,N,Mdm,Mzl,bsample,target] for i in range(N)])

    bary_file_name = whereto + samplefile[:-5]+'_barys_%s.json'%id
    json.dump(results, open(bary_file_name,'w'),indent = 2)
    
    print (f'Total work completed in {(time.time()-starttime)/60.} mins')





















































