import matplotlib.pyplot as plt
import numpy as np
import sys,json,os
sys.path.append("..")
import project.tools.statistics as stat1
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def get_exlims(exclfiles):
    """ getting the various excluison limits from the result file of excllim
    -----------
    exclfiles: filenames, either a list of filenames in work_dir/Output or a 
               str of a filename in work_dir/Output
    returns the Mdm values, higher and lower constructive limits, higher and 
            lower destructive limits.
    ** This program is made for "binned" exclusion limit calculation in "dmgl" plane only.
    """
    if isinstance(exclfiles, str):
        exclfiles = [exclfiles]
    order = []
    glh_c = []
    gll_c = []
    glh_d = []
    gll_d = []
    Mdm = []
    for exfile in exclfiles:
        samplepath = '../Output/%s'%exfile
        inputdata = json.load(open(samplepath))

        order_ = np.array(inputdata['gl order'])
        glh_c_ = np.array(inputdata['gl high constructive'])
        gll_c_ = np.array(inputdata['gl low constructive'])
        glh_d_ = np.array(inputdata['gl high destructive'])
        gll_d_ = np.array(inputdata['gl low destructive'])
        gll_d_[gll_d_ == 0.0001] = np.NaN
        gll_c_[gll_c_ == 0.0001] = np.NaN

        glh_c.append(glh_c_*order_)
        gll_c.append(gll_c_*order_)
        glh_d.append(glh_d_*order_)
        gll_d.append(gll_d_*order_)
        Mdm.append(inputdata['DM mass array'])
    if len(exclfiles) != 1:
        return Mdm, glh_c, gll_c, glh_d, gll_d
    else:
        return Mdm[0], glh_c[0], gll_c[0], glh_d[0], gll_d[0]
    
import matplotlib.colors as colors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

    
Brbg = truncate_colormap(plt.get_cmap('BrBG'),0,1)
Br = truncate_colormap(plt.get_cmap('BrBG'),0,0.43)
Bg = truncate_colormap(plt.get_cmap('BrBG'),0.67,1.)
Br_r = truncate_colormap(plt.get_cmap('BrBG'),0.43,0)
Bg_r = truncate_colormap(plt.get_cmap('BrBG'),1.,0.67)
br = plt.get_cmap('BrBG')(0.2)
bg = plt.get_cmap('BrBG')(0.8)
    
class Analyze:
    def __init__(self, Res, **kwargs):
        """
        Class for analyzing the results as given out by the reconstruction 
        code and storen in json file: which is to be read and passed to this 
        class as res.
        If there are more than one res.. give res as [res_1, res_2, ... res_n_res] -
        - In this case.. N should be same for all the res's.
        kwargs::
        N: number of grid points of Mdm and Mzl where reconstruction is done
        nuis_params: nuisance parameters in reconstruction. Should be given in the
                     order in which it appears in res file
        fixed_params: fixed parametes in reconstruction should be given as
                      a dict, with keys as parameter names and the fixed value.
                      eg: {'gh':1e-15}
        res_keys: keys for each of res in Res
        """
        
        self.res = Res
        
        if 'N' in kwargs:
            self.N = kwargs.get('N')
        else:
            self.N = 40
            
        if len(self.res) != self.N:
            if len(self.res[0]) == self.N:
                self.n_res = len(self.res)  
            else:
                print (f'len(Res) = {len(self.res)}, N = {self.N}, len(Res[0]) = {len(self.res[0])}:\n' +
                     'therefore we assume there are more than one res \n' +
                     'If not true.. make the change now.')
        else:
            self.n_res = 1
            self.res = [self.res]
            
        if 'nuis_params' in kwargs:
            self.nuis_params = kwargs.get('nuis_params')
        else:
            self.nuis_params = ['bl','gl','gh']
            
        if 'fixed_params' in kwargs:
            self.fixed = True
            self.fixed_params = kwargs.get('fixed_params')
        else:
            self.fixed = False
            self.fixed_params = []
        
        self.res_dm = np.zeros([self.N,self.N])
        self.res_zl = np.zeros([self.N,self.N])
        self.res_ll = np.zeros([self.N,self.N])
        
        self.res_params = {}
        for p in ['dm','zl','ll',*self.nuis_params]:
            self.res_params[p] = [np.zeros([self.N,self.N]) for i in range(self.n_res)]
        
        for l in range(self.n_res):
            res = self.res[l]
            for i in range(self.N):
                for j in range(self.N):
                    k = 0
                    for key in self.res_params.keys():
                        k += 1
                        self.res_params[key][l][i,j] = res[i][k][j] 
                    
        self.res_params['ll_best'] = [[] for i in range(self.n_res)]
        self.res_params['test'] = [[] for i in range(self.n_res)]
        self.best_dm_zl = [[] for i in range(self.n_res)]
        for l in range(self.n_res):
            i,j = np.argwhere(self.res_params['ll'][l] == np.min(self.res_params['ll'][l]))[0]
            self.res_params['ll_best'][l] = self.res_params['ll'][l][i,j]
            self.res_params['test'][l] = 2.*(self.res_params['ll'][l] - self.res_params['ll_best'][l])
            self.best_dm_zl[l] = [self.res_params['dm'][l][i,j],self.res_params['zl'][l][i,j]]
                    
                
        if 'res_keys' in kwargs:
            self.res_keys = kwargs.get('res_keys')
        else:
            self.res_keys = None
            
    def plot_CI(self, ax = None, ts = [5.99,9.2], cmaps = [Bg,Br], CI = [95,99],
                clrbar = True, xlim = [1,10], ylim = [1,20], plt_cf = True, **kwargs):
        """
        cmaps: len(cmaps) = len(ts),  colors for the confidence intervals in contour plot
               will be be derived from cmaps
        CI: CI is the corresponding COnfidence interval to the ts
        """
        if isinstance(ts, float):
            ts = [ts]
            
        if len(cmaps) >= len(ts):
            cmaps = cmaps[:len(ts)]
        else:
            print ('Please provide len(cmaps) == len(ts)')
            return None
        
        colors = [[] for i in range(self.n_res)]
        l = -1
        for c_indx in np.linspace(0.,0.99,self.n_res+1,endpoint = False)[1:]:
            l += 1
            colors[l] = []
            for i in range(len(ts)):
                cmap = cmaps[i]
                colors[l].append(cmap(c_indx))
        clrs = [colors[-(i + 1)] for i in range(len(colors))]
        
        if not ax:
            fig,ax = plt.subplots(1,1,figsize = (6,4))
            
        for l in range(self.n_res):
            if plt_cf:
                cn  = ax.contourf(self.res_params['dm'][l], self.res_params['zl'][l], 
                                  self.res_params['test'][l],levels = [0,*ts], 
                                  colors = colors[l],**kwargs)
            clrs = [colors[l][-(i+1)] for i in range(len(colors[l]))]
            ax.contour(self.res_params['dm'][l], self.res_params['zl'][l], 
                       self.res_params['test'][l], levels = [0,*ts], 
                       colors = clrs,**kwargs)
            ax.scatter(self.best_dm_zl[l][0],self.best_dm_zl[l][1],marker = '*', c = 'r', s = 60)
        if not len(ts) == 1:
            if clrbar:
                cb = plt.colorbar(cn,ax = ax)
                cb.set_label(label = '$t_{\mathbf{\mu}}$',size = 12,weight = 'bold')
                cb.set_ticks(ts)
                cb.set_ticklabels(['%.1f (%i%%)'%(ts[i],CI[i]) for i in range(len(ts))])
        else:
            ax.text(.81,.93,'%i%% (%.2f)'%(CI[0],ts[0]),transform = ax.transAxes)
        ax.set_ylabel('$\mathrm{m_{Z_l}}$ (GeV)',size = 14)
        ax.set_xlabel('$\mathrm{m_{dm}}$ (GeV)',size = 14)
        
        if self.res_keys:
            if len(self.res_keys) != self.n_res:
                print ('length of res_keys and n_res do not match')
            else:
                for l in range(self.n_res):
                    if self.res_keys[l]:
                        ax.plot([-1e4,-1e4],[1e-4,1e-4],c = colors[l][0], label = self.res_keys[l])
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.legend(loc = 'lower right')
                
    def plot_param(self, param, ax = None, levels = None, cmap = Bg, label = None):
        
        if not ax:
            fig,ax = plt.subplots(1,1,figsize = (6,4))
            
        if isinstance(param, str):
            z = self.res_params[param][0]
        else:
            if isinstance(param, list):
                z = param[1](self.res_params[param[0]][0])
            else:
                print ('please give param as a string or a list with param as the first element' +
                        'and the function using that param as the 2nd element')
            
            
            
        if levels:
            for l in range(self.n_res):
                cn = ax.contourf(self.res_params['dm'][l], self.res_params['zl'][l], 
                                 z, cmap = cmap, alpha = 0.7, levels = levels)
        else:
            for l in range(self.n_res):
                cn = ax.contourf(self.res_params['dm'][l], self.res_params['zl'][l],
                                 z, cmap = cmap, alpha = 0.7)
        
        cb = plt.colorbar(cn, ax = ax)
        if label:
            cb.set_label(label = label)
        else:
            if not isinstance(param, str):
                label = param[0]
            else:
                label = param
            cb.set_label(label = label)
        ax.set_ylabel('$\mathrm{m_{Z_l}}$ (GeV)',size = 14)
        ax.set_xlabel('$\mathrm{m_{dm}}$ (GeV)',size = 14)

def plot_all(ft_dm,ft_zl,ft_ll,ft_ps,ps_levels,fit_id,parent_file,
             actual,sample,Nbins,target,eary,btot,B0,B1,B2,bbg,levels = [0.,5.99,10.]):
	"""
	ft_ps = fit_parameters should be a dict with key word the key of that parameter and the 
	minimized values as the dict content eg {'bl':bl_bl, 'gl': gl_gl}.
	ps_levels is also a dict
	"""
	
	i1,j1 = np.argwhere(ft_ll == np.min(ft_ll))[0]
	ft_best = ft_ll[i1,j1]
	ft_test = 2.*(ft_ll - ft_best)
	print ('fig1')
	
	fig,ax = plt.subplots(figsize = (6,6))
	c1 = ax.contourf(ft_dm,ft_zl,ft_test, levels = levels)
	ax.contour(ft_dm,ft_zl,ft_test,levels = levels)
	
	ax.set_xlabel('$m_{DM}$ in GeV',size = 15)
	ax.set_ylabel('$m_{Z_l}$ in MeV',size = 15)
	
	ax.scatter(ft_dm[i1,j1],ft_zl[i1,j1],marker = '*',c='r',s=200,label = 'Best fit')
	ax.scatter(actual['mdm'],actual['mzl'],marker = '*', c='k',s=200,label = 'Actual model')
	
	ax.legend()
	
	fig.colorbar(c1, label = 'test statistics')
	key = ''
	
	for keys in ft_ps:
		key += keys
	
	plt.savefig('%s/%s/test_stat%s.png'%(parent_file,key,fit_id),bbox_inches = 'tight')
	
	print ('fig2')
	
	for keys in ft_ps:
		ps = ft_ps[keys]
		#if keys == 'gl':
		#	ps = -np.log10(ft_ps[keys])
		#if keys == 'gh':
		#	ps = -np.log10(np.abs(ft_ps[keys]))
			
		fig,ax = plt.subplots(figsize = (6,6))
		c2 = ax.contourf(ft_dm, ft_zl, ps, levels = ps_levels[keys])
		ax.contourf(ft_dm,ft_zl,ps,levels = ps_levels[keys])
		ax.contour(ft_dm,ft_zl,ft_test,levels = [0,5.99],colors = ['blue'])
		
		ax.scatter(ft_dm[i1,j1],ft_zl[i1,j1],marker = '*',c='r',s=200,label = 'Best fit')
		ax.scatter(actual['mdm'],actual['mzl'],marker = '*',c='k',s=200,label = 'Actual model')
		
		if keys == 'gl' or keys == 'gh':
			label = '-log(%s)'%keys
		else:
			label = keys
		
		fig.colorbar(c2,label = label)
		
		ax.set_xlabel('$m_{DM}$ in GeV', size = 15)
		ax.set_ylabel('$m_{Z_l}$ in MeV',size = 15)
	
		plt.savefig('%s/%s/contours_%s_%s.png'%(parent_file,key,keys,fit_id),bbox_inches = 'tight')
		
	for key1 in ft_ps:
		if key1 == 'gl':
			take_gl = 10**-ft_ps['gl'][i1,j1]
		else:
			take_gl = actual['gl']
		
		if key1 == 'gh':
			take_gh = 10**-fit_ps['gh'][i1,j1]
			print ('specify interference.. wrong...')
		else:
			take_gh = actual['gh']
		
		if key1 == 'bl':
			take_bl = fit_ps['bl'][i1,j1]
		else:
			take_bl = actual['bl']
			
	print ('fig3')
	fig,ax = plt.subplots(figsize = (6,6))
	ax.hist(sample,Nbins,histtype = 'step',color = 'k',label = 'Actual model')
	ax.plot(eary,btot,ls='--',c='k',lw = 3.,label = 'Actual')
	btot_a = stat1.binned.bintot(target,ft_dm[i1,j1],ft_zl[i1,j1],take_gl,take_gh,B0[i1,j1,:,:],
	B1[i1,j1,:,:],B2[i1,j1,:,:],bbg,take_bl)
	ax.plot(eary,btot_a,ls = '-',c='r',lw = 2.,label = 'Best fit')
	
	ax.set_yscale('log')
	ax.set_xscale('log')
	
	ax.set_xlabel('$E_{recoil}$ in KeV', size = 15)
	ax.set_ylabel('Counts',size = 15)
	
	ax.legend()
	
	plt.savefig('%s/%s/rate_%s'%(parent_file,key,fit_id),bbox_inches = 'tight')
	
	print ('fig4')
	fig,ax = plt.subplots(figsize = (6,6))
	c3 = ax.contourf(ft_dm,ft_zl,ft_ll,levels = np.linspace(ft_best,ft_best + 200,40))
	fig.colorbar(c3, label = 'log-likelihood')
	ax.set_xlabel('$m_{DM}$ in GeV',size = 15)
	ax.set_ylabel('$m_{Z_l}$ in MeV',size = 15)
	plt.savefig('%s/%s/llike_%s'%(parent_file,key,fit_id),bbox_inches = 'tight')	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
from matplotlib.collections import LineCollection

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

