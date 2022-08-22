import numpy as np
import matplotlib.pyplot as plt
import json
import time
from mpi4py import MPI

import sys
sys.path.append('..')
import project.tools.statistics as stat1
from project.plot_ll import plot_all as pall
import project.reconst_funcs as rf

def PllMini(model,M,N,paranames,guess0s):
	"""
	model is a string eg. 'I'
	for the 10 models it can be any of Model = ['I','II','III','IV','V','VI','VII','VIII','IX','X']
	model is not limited to this list.
	"""
	
	inputdata = json.load(open('../Mock_data/NaI_Model%s_mock.json'%model,'r'))
	
	targetdata = inputdata['target']
	material = targetdata['material']
	exp = targetdata['exposure']
	thr = targetdata['thr']
	res = targetdata['sigma']
	roi = targetdata['roi']
	bg  = targetdata['bg']
	target = stat1.target(*material,exposure=exp,thr=thr,sigma=res,bg = bg,roi=roi)
	
	
	print ('1/4 done')
	
	modeldata = inputdata['model']
	actual = {'mdm': modeldata['mdm'],
		      'mzl': modeldata['mZl'],
		      'gl' : modeldata['gl'],
		      'gh' : modeldata['gheff'],
		      'bl' : bg[0],
		     }

	sample = np.array(inputdata['sample'])
	
	Nbins = int(39./(15*res))
	fin = int(((roi - thr)/float(Nbins)/res))
	bsample = stat1.binned.binsample(sample,Nbins,fin)
	bbg = stat1.binned.bintbg(target,bsample)
	
	b0,b1,b2 = rf.b_arys(bsample,target,actual['mdm'],actual['mzl'])
	eary = rf.e_arys(bsample,Nbins)
	btot = stat1.binned.bintot(target,actual['mdm'],actual['mzl'],actual['gl'],actual['gh'],
		   b0,b1,b2,bbg,actual['bl'])

	print ('2/4 done')

	if not isinstance(paranames, (list,np.ndarray)):
		paranames = [paranames]
		guess0s = [guess0s]

	stt = time.time()
	Mdm = np.linspace(1.,10.,M)
	Mzl = np.linspace(1.,20.,N)
	
	barys = json.load(open('barys_%s%s_model%s.json'%(M,N,model),'r'))
	B0 = np.array(barys['b0'])
	B1 = np.array(barys['b1'])
	B2 = np.array(barys['b2'])
	
	print ('3/4 done')
		
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	
	count = M // size
	reminder = M % size
	
	print ('4/4 done')
	if rank < reminder:
		start = rank*(count + 1)
		stop  = start + count + 1
	else:
		start = rank * count + reminder
		stop = start + count 
	
	local_Mdm = Mdm[start:stop]
	local_b0 = B0[start:stop,:,:,:]
	local_b1 = B1[start:stop,:,:,:]
	local_b2 = B2[start:stop,:,:,:]
		
	# Change the function here manually
	local_results = rf.reconstruct_gl(local_Mdm,Mzl,guess0s[0],local_b0,local_b1,local_b2,stt,
	bsample,target,bbg,actual)
	#local_dm,local_zl,local_ll,local_gl = local_results
	local_results = np.array(local_results)
	
	paralength = len(paranames)
	if (len(local_results[3]) != paralength):
		print (f'len of local_results[3] {len(local_results[3])} != paralength {paralength}')
	
	if rank > 0:
		comm.Send(local_results, dest = 0, tag = 14)
	else:
		final_dm = np.copy(local_results[0])
		final_zl = np.copy(local_results[1])
		final_ll = np.copy(local_results[2])
		final_gl = np.copy(local_results[3])
		
		for i in range(1,size):
			if i < reminder:
				rank_size = count + 1
			else:
				rank_size = count
			tmp = np.empty((4,rank_size,N),dtype = float)
			comm.Recv(tmp,source = i, tag = 14)
			final_dm = np.vstack((final_dm,tmp[0]))
			final_zl = np.vstack((final_zl,tmp[1]))
			final_ll = np.vstack((final_ll,tmp[2]))
			final_gl = np.vstack((final_gl,tmp[3]))
			
		edt	= time.time()
		print (f'Total time take is {(edt-stt)//60} min {(edt-stt)%60} sec.')
		print (final_dm.shape,final_gl.shape)
		
		
		gldict = {'dm':final_dm.tolist(),
				  'zl':final_zl.tolist(),
				  'll':final_ll.tolist(),
				  'gh':final_gl.tolist()}
		filename = ''
		for ps in paranames:
			if ps == 'nlgl':
				ps = 'gl'
			elif ps == 'nlgh':
				ps = 'gh'
			filename += ps
		
		parent_file = '../Output/dmzl2/%s/'%filename
		json.dump(gldict,open('%s/Model$s_psdict.json'%parent_file,'w'),indent = 2)
		
		print ('saved gldict with keys')	
		for keys in gldict:
			print ('{keys}, ')
			
		# Make change here also.	
		#rf.plotting(final_dm,final_zl,final_ll,final_Ps[0],model)
		
		gl_levels = [5,6,7,7.5,8,8.5,9,9.2,9.4,9.6,9.8,10.,10.2,10.4,10.6,11.,12.,13.,14.,15.]
		pall(final_dm,final_zl,final_ll,{'gl':final_gl},{'gl':gl_levels},
			fit_id = model,
			parent_file = '../Output/dmzl2',
			actual = actual,
			sample = sample,
			Nbins = Nbins,
			target = target,
			eary = eary,
			btot = btot,
			B0 = B0, B1 = B1, B2 = B2,
			bbg = bbg)
		
		print (f'Total time take is {(time.time()-stt)//60} min {(time.time()-stt)%60} sec.')
		
