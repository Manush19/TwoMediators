import argparse, sys, json, os, math
import numpy as np
import scipy as sp
from scipy import optimize
from tqdm import tqdm
sys.path.append("..")
sys.path.append('../../../')
import project.tools.statistics as stat
from project.tools.bisect import rbisection, lbisection
from project.tools.quadrant_hopping import quadranthop
from project.constants import cosmo
from project.tools import magnitude

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

# dmhelp = ("Value for DM mass for which the exclusion limits should be found "
#             +"If DM is in plane, this values sets the lower limit of the "
#             +"the logarithmically spaced array of DM masses on the x-axis, i.e. np.logspace(>input<,1).")
# funchelp = ("This program reads in a .json sample file from the output folder as created by sample.py and finds "
#             +"the exclusion limits for a predefined plane of the free parameters.")

# parser = argparse.ArgumentParser(description=funchelp)
# group = parser.add_mutually_exclusive_group()
# group.add_argument("-b", "--binned", type=int, help="Binned sample, enter number of bins.")
# group.add_argument("-u", "--unbinned", action="store_true", help='Unbinned sample.')
# parser.add_argument("sample", help="Name of sample .json file in output folder.")
# parser.add_argument("plane", choices=['dmgl','mZgl','dmgh','glgh','dmmZ'], help="Parameter-plane in which the exclusion limits should be found, choose from the above.")
# parser.add_argument("-dm", type=float, default=-1, help=dmhelp)
# parser.add_argument("-mZ", type=float, help="Value for mediator mass for which the exclusion limits should be found (will be ignored if mZ in param).")
# parser.add_argument("-gl", type=float, help="Value for light mediator coupling for which the exclusion limits should be found (will be ignored if gl in param).")
# parser.add_argument("-gh", type=float, help="Value for effective heavy mediator coupling for which the exclusion limits should be found (will be ignored if gh in param).")
# parser.add_argument("-n", default=100, type=int, help="For how many values on the x-axis the excl. limit is evaluated.")
# parser.add_argument("-p", "--pvalue", default=0.05, type=float)
# parser.add_argument("-id", help="Give an additional identification to the name of the resulting .json file.")
# args = parser.parse_args()

def Tqdm(show_tqdm, n):
    if show_tqdm:
        return tqdm(range(n))
    else:
        return range(n)

def excl(binned,sample, id, plane = 'dmgl', mZ = 1, gl = 1e-10, gh = 1e-10, dm = -1,n = 50 ,p = 0.05, fin = None, show_tqdm = False):
    """ For obtaining the exclusion limit for a given sample file in the specified plane
    ------------------
    binned: 'To find exlusion limits in binned sample. Give the bin number. To do 
             unbinned sample analysis, pass binned = 0.
    sample: Name of sample .json file in the Output folder
    id: give an additional identification to the name of the resulting .json file
    plane: Parameter-plane in which the exclusion limits should be found, choose from 
           ('dmgl', 'mZgl', 'dmgh', 'glgh', 'dmmZ')
    mZ: Value for mediator mass for which the excluson limits should be found (will be 
        ignored if mzl in the parameter-plane"
    gl: Value for light mediator coupling for which the exclusion limits should be found
        (will be ignored if gl in param)
    gh: Value for effective heavy mediator coupling (will be ignored if gh in plane)
    n: For how many values on the x-axis the excl. limit is evaluated (defaul = 50)
    p: pvalue to be used. default = 0.05 (95% exclusion limts)
    fin: Number of enery points in each bin to evaluate the data. (accuracy)
         (default = 5) if fin == None, it is estimated from the bin and resolution.
    show_tqdm = True for showing the progress of the code running. 
    """
    class Arguments:
        def __init__(self,binned,sample,plane,dm,mZ,gl,gh,n,p,id,fin):
            self.binned = binned
            self.sample = sample
            self.plane = plane
            self.dm = dm
            self.mZ = mZ
            self.gl = gl
            self.gh = gh
            self.n = n
            self.pvalue = p
            self.id = id
            self.fin = fin
        
    args = Arguments(binned,sample,plane,dm,mZ,gl,gh,n,p,id,fin)

    n = args.n
    chi = sp.stats.chi2.ppf(1-args.pvalue,1)

    whereto = os.path.join(os.getcwd(), "../Output")
    if not os.path.isdir(whereto):
        print ("No output directory found")
        return None
    if os.access(whereto, os.R_OK):
        samplepath = os.path.join(whereto, args.sample)
        if not os.access(samplepath, os.F_OK):
            print("Sample file not found")
            return None
        if os.access(samplepath, os.F_OK):
            with open(samplepath) as json_file: 
                inputdata = json.load(json_file)
            json_file.close() 
        else:
            print ("readable_dir:{0} is not a readable dir/file".format(samplepath))
            return None
    else:
        print ("readable_dir:{0} is not a readable dir".format(whereto))
        return None

    # read out target data from file
    targetdata = inputdata['target']
    material = targetdata['material']
    exp = targetdata['exposure']
    thr = targetdata['thr']
    res = targetdata['sigma']
    bg = targetdata['bg']
    roi = targetdata['roi']
    target = stat.target(*material, exposure = exp, thr = thr, sigma = res, bg = bg, roi = roi)
    
    vdf = np.array(inputdata['vdf'])
    ve = np.array(inputdata['ve'])
    vesc = inputdata['vesc']
    vcir = inputdata['vcir']
    I_index = inputdata['I_index']
    if vdf[0] == 'SHM':
        target.initialize_vdf(I_index = I_index,vesc = vesc, vcir = vcir)
    else:
        target.initialize_vdf(I_index = 2, VDF = vdf, V = ve, vesc = vesc, vcir = vcir)

    # read out model data from file
    modeldata = inputdata['model']
    m_dm = modeldata['mdm']
    m_Z_l = modeldata['mZl']
    g_l = modeldata['gl']
    g_heff = modeldata['gheff']

    # read out sample from file
    sample = np.array(inputdata['sample'])

    if args.binned:
        bins = args.binned
        if args.fin:
            fin = args.fin
        else:
            if res != 0:
                fin = int(((roi-thr)/bins)/res)
            else:
                fin = int(((roi-thr)/bins)/0.1)
        bsample = stat.binned.binsample(sample, bins, fin)
        bbg = stat.binned.bintbg(target, bsample)
        N_sample = bsample['Ntot']

        if args.plane == "dmgl":
            mdm_array = np.logspace(args.dm,1,n)
            binnum = bsample['binnum']
            order = np.zeros(n)
            # in the first column is always the constructive, in the second column
            # the destructive case
            mlgl = np.zeros([n,2])
            mlfun = np.zeros([n,2])
            mlNsig = np.zeros([n,2])
            limgl_h = np.zeros([n,2])
            limgl_l = np.zeros([n,2])
            # first index for mass, second index for bin number, third index for target element
            b0_array = np.zeros([n,binnum,target.nelements])
            b1_array = np.zeros([n,binnum,target.nelements])
            b2_array = np.zeros([n,binnum,target.nelements])

            mZl = args.mZ
            gh = args.gh

            for i in Tqdm(show_tqdm,n):
                Emax = target.E_max(mdm_array[i])
                if Emax > (thr-2*res):
                    order[i] = stat.getglorder(N_sample,target,mdm_array[i],mZl)
                    b0_array[i,:,:] = stat.binned.bint0(target,bsample,mdm_array[i])
                    b1_array[i,:,:] = stat.binned.bint1(target,bsample,mdm_array[i],mZl)
                    b2_array[i,:,:] = stat.binned.bint2(target,bsample,mdm_array[i],mZl)

                    # finding best ML estimate
                    asimovres = optimize.minimize(stat.binned.llbl,1,args=(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,target,mdm_array[i],mZl,0,gh,order[i]))
                    asimov = asimovres.fun
                    const = quadranthop(stat.binned.llglpos,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],
                                            bbg,target,mdm_array[i],mZl,gh,order[i]),('pe','pe'))
                    dest = quadranthop(stat.binned.llglpos,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],
                                            bbg,target,mdm_array[i],mZl,-gh,order[i]),('pe','pe'))
                    if const.success == False or const.globalfun>=asimov:
                        mlgl[i,0] = 0
                        mlfun[i,0] = asimov
                        mlNsig[i,0] = np.sum(stat.binned.bintot(target,mdm_array[i],mZl,0,gh,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],0*bbg,1))
                    else:
                        mlgl[i,0] = const.globalmin[1]
                        mlfun[i,0] = const.globalfun
                        mlNsig[i,0] = np.sum(stat.binned.bintot(target,mdm_array[i],mZl,mlgl[i,0]*order[i],gh,
                                                                b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],0*bbg,1))

                    if dest.success == False or dest.globalfun>=asimov:
                        mlgl[i,1] = 0
                        mlfun[i,1] = asimov
                        mlNsig[i,1] = mlNsig[i,0]
                    else:
                        mlgl[i,1] = dest.globalmin[1]
                        mlfun[i,1] = dest.globalfun
                        mlNsig[i,1] = np.sum(stat.binned.bintot(target,mdm_array[i],mZl,mlgl[i,1]*order[i],-gh,
                                                                b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],0*bbg,1))
                else:
                    mlgl[i,0] = 0
                    mlgl[i,1] = 0

            if np.count_nonzero(mlgl[:,0])<0.2*n:
                mlgl[:,0] = np.zeros(n)

            for i in Tqdm(show_tqdm,n):
                # find exclusion limits
                # constructive
                Emax = target.E_max(mdm_array[i])
                if Emax > (thr-2*res):
                    if mlgl[i,0] == 0:
                        rbic = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                                target,mlfun[i,0],mlNsig[i,0],chi,mdm_array[i],mZl,gh,order[i]),
                                        1e-3,1e4,sep=1.001)
                        if rbic.success == True:
                            limgl_h[i,0] = rbic.x0
                            limgl_l[i,0] = np.NaN
                        else:
                            limgl_h[i,0] = np.NaN
                            limgl_l[i,0] = np.NaN
                    else:
                        rbic = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                                target,mlfun[i,0],mlNsig[i,0],chi,mdm_array[i],mZl,gh,order[i]),
                                        mlgl[i,0],1e4,sep=1.001)
                        if rbic.success == True:
                            limgl_h[i,0] = rbic.x0
                        else:
                            limgl_h[i,0] = np.NaN
                        limgl_l[i,0] = np.NaN

                    # destructive
                    if mlgl[i,1] == 0:
                        rbid = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                                target,mlfun[i,1],mlNsig[i,1],chi,mdm_array[i],mZl,-gh,order[i]),
                                        1e-3,1e4,sep=1.001)
                        if rbid.success == True:
                            limgl_h[i,1] = rbid.x0
                            limgl_l[i,1] = np.NaN
                        else:
                            limgl_h[i,1] = np.NaN
                            limgl_l[i,1] = np.NaN
                    else:
                        rbid = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                                target,mlfun[i,1],mlNsig[i,1],chi,mdm_array[i],mZl,-gh,order[i]),
                                        mlgl[i,1],1e4,sep=1.001)
                        lbid = lbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                                target,mlfun[i,1],mlNsig[i,1],chi,mdm_array[i],mZl,-gh,order[i]),
                                        1e-4,mlgl[i,1],sep=1.001)
                        if rbic.success == True:
                            limgl_h[i,1] = rbid.x0
                            limgl_l[i,1] = lbid.x0
                        else:
                            limgl_h[i,1] = np.NaN
                            limgl_l[i,1] = np.NaN
                else:
                    limgl_h[i,0] = np.NaN
                    limgl_h[i,1] = np.NaN
                    limgl_l[i,0] = np.NaN
                    limgl_l[i,1] = np.NaN

            outputdict = {"DM mass array" : list(mdm_array),
                            "med. mass" : mZl,
                            "gh" : gh,
                            "gl order" : list(order),
                            "gl ML constructive" : list(mlgl[:,0]),
                            "gl ML destructive" : list(mlgl[:,1]),
                            "gl high constructive" : list(limgl_h[:,0]),
                            "gl low constructive" : list(limgl_l[:,0]),
                            "gl high destructive" : list(limgl_h[:,1]),
                            "gl low destructive" : list(limgl_l[:,1]),
                            }

            split_ = args.sample.split('_')
            targetname = split_[0]
            targetid = split_[-1].split('.')[0]
            if args.id:
                fileid = args.id
                filename = args.plane + "_" + targetname + "_" + targetid + "_" + fileid + ".json"
            else:
                filename = args.plane + "_" + targetname + "_" + targetid + ".json"



            whereto_out = os.path.join(whereto, filename)

            with open(whereto_out, 'w') as fp:
                json.dump(outputdict, fp,  indent=4)
            fp.close


        elif args.plane == "mZgl":
            mZ_array = np.logspace(-1,2,n)
            order = np.zeros(n)
            binnum = bsample['binnum']
            # in the first column is always the constructive, in the second column
            # the destructive case
            mlgl = np.zeros([n,2])
            mlfun = np.zeros([n,2])
            mlNsig = np.zeros([n,2])
            limgl_h = np.zeros([n,2])
            limgl_l = np.zeros([n,2])
            # first index for mass, second index for bin number, third index for target element
            b0_array = np.zeros([n,binnum,target.nelements])
            b1_array = np.zeros([n,binnum,target.nelements])
            b2_array = np.zeros([n,binnum,target.nelements])

            mdm = args.dm
            gh = args.gh

            for i in tqdm(range(n)):
                order[i] = stat.getglorder(N_sample,target,mdm,mZ_array[i])
                b0_array[i,:,:] = stat.binned.bint0(target,bsample,mdm)
                b1_array[i,:,:] = stat.binned.bint1(target,bsample,mdm,mZ_array[i])
                b2_array[i,:,:] = stat.binned.bint2(target,bsample,mdm,mZ_array[i])

                # finding best ML estimate
                asimovres = optimize.minimize(stat.binned.llbl,1,args=(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,target,mdm,mZ_array[i],0,gh,order[i]))
                asimov = asimovres.fun
                const = quadranthop(stat.binned.llglpos,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],
                                    bbg,target,mdm,mZ_array[i],gh,order[i]),('pe','pe'))
                dest = quadranthop(stat.binned.llglpos,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],
                                    bbg,target,mdm,mZ_array[i],-gh,order[i]),('pe','pe'))
                if const.success == False or const.globalfun>=asimov:
                    mlgl[i,0] = 0
                    mlfun[i,0] = asimov
                    mlNsig[i,0] = np.sum(stat.binned.bintot(target,mdm,mZ_array[i],0,gh,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],0*bbg,1))
                else:
                    mlgl[i,0] = const.globalmin[1]
                    mlfun[i,0] = const.globalfun
                    mlNsig[i,0] = np.sum(stat.binned.bintot(target,mdm,mZ_array[i],mlgl[i,0]*order[i],gh,
                                                            b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],0*bbg,1))
                if dest.success == False or dest.globalfun>=asimov:
                    mlgl[i,1] = 0
                    mlfun[i,1] = asimov
                    mlNsig[i,1] = mlNsig[i,0]
                else:
                    mlgl[i,1] = dest.globalmin[1]
                    mlfun[i,1] = dest.globalfun
                    mlNsig[i,1] = np.sum(stat.binned.bintot(target,mdm,mZ_array[i],mlgl[i,1]*order[i],-gh,
                                                            b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],0*bbg,1))

            if np.count_nonzero(mlgl[:,0])<0.2*n:
                mlgl[:,0] = np.zeros(n)

            for i in tqdm(range(n)):
                # find exclusion limits
                # constructive
                if mlgl[i,0] == 0:
                    rbic = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                              target,mlfun[i,0],mlNsig[i,0],chi,mdm,mZ_array[i],gh,order[i]),
                                    1e-5,1e4,sep=1.001)
                    if rbic.success == True:
                        limgl_h[i,0] = rbic.x0
                        limgl_l[i,0] = np.NaN
                    else:
                        limgl_h[i,0] = np.NaN
                        limgl_l[i,0] = np.NaN
                else:
                    rbic = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                              target,mlfun[i,0],mlNsig[i,0],chi,mdm,mZ_array[i],gh,order[i]),
                                    mlgl[i,0],1e4,sep=1.001)
                    if rbic.success == True:
                        limgl_h[i,0] = rbic.x0
                    else:
                        limgl_h[i,0] = np.NaN
                    limgl_l[i,0] = np.NaN

                # destructive
                if mlgl[i,1] == 0:
                    rbid = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                              target,mlfun[i,1],mlNsig[i,1],chi,mdm,mZ_array[i],-gh,order[i]),
                                    1e-5,1e4,sep=1.001)
                    if rbid.success == True:
                        limgl_h[i,1] = rbid.x0
                        limgl_l[i,1] = np.NaN
                    else:
                        limgl_h[i,1] = np.NaN
                        limgl_l[i,1] = np.NaN
                else:
                    rbid = rbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                              target,mlfun[i,1],mlNsig[i,1],chi,mdm,mZ_array[i],-gh,order[i]),
                                    mlgl[i,1],1e4,sep=1.001)
                    lbid = lbisection(stat.binned.ts_excl_gl,(bsample,b0_array[i,:,:],b1_array[i,:,:],b2_array[i,:,:],bbg,
                                                              target,mlfun[i,1],mlNsig[i,1],chi,mdm,mZ_array[i],-gh,order[i]),
                                    1e-3,mlgl[i,1],sep=1.001)
                    if rbic.success == True:
                        limgl_h[i,1] = rbid.x0
                        limgl_l[i,1] = lbid.x0
                    else:
                        limgl_h[i,1] = np.NaN
                        limgl_l[i,1] = np.NaN

            outputdict = {  "dm mass" : mdm,
                            "gh" : gh,
                            "gl order" : list(order),
                            "gl ML constructive" : list(mlgl[:,0]),
                            "gl ML destructive" : list(mlgl[:,1]),
                            "gl high constructive" : list(limgl_h[:,0]),
                            "gl low constructive" : list(limgl_l[:,0]),
                            "gl high destructive" : list(limgl_h[:,1]),
                            "gl low destructive" : list(limgl_l[:,1]),
                            }

            split_ = args.sample.split('_')
            targetname = split_[0]
            targetid = split_[-1].split('.')[0]
            if args.id:
                fileid = args.id
                filename = args.plane + "_" + targetname + "_" + targetid + "_" + fileid + ".json"
            else:
                filename = args.plane + "_" + targetname + "_" + targetid + ".json"



            whereto_out = os.path.join(whereto, filename)

            with open(whereto_out, 'w') as fp:
                json.dump(outputdict, fp,  indent=4)
            fp.close

        else: 
            raise NameError('Exclusion plane not yet implemented')           

  
    else: # unbinned loglikelihood:  
        N_sample = len(sample) 
        ibg = target.bgintegral(target.roi)
        bg_array = target.bg(sample)

        if args.plane == "dmgl":
            mdm_array = np.logspace(args.dm,1,n)
            order = np.zeros(n)
            # in the first column is always the constructive, in the second column
            # the destructive case
            mlgl = np.zeros([n,2])
            mlfun = np.zeros([n,2])
            mlNsig = np.zeros([n,2])
            limgl_h = np.zeros([n,2])
            limgl_l = np.zeros([n,2])
            # first index for mass, second index for target element
            i0_array = np.zeros([n,target.nelements])
            i1_array = np.zeros([n,target.nelements])
            i2_array = np.zeros([n,target.nelements])
            # first index for mass, second index for target element, third for the sample member
            E0_array = np.zeros([n,target.nelements,N_sample])
            E1_array = np.zeros([n,target.nelements,N_sample])
            E2_array = np.zeros([n,target.nelements,N_sample])

            mZl = args.mZ
            gh = args.gh

            for i in tqdm(range(n)):
                Emax = target.E_max(mdm_array[i])
                if Emax > (thr-2*res):
                    order[i] = stat.getglorder(N_sample,target,mdm_array[i],mZl)
                    i0_array[i,:] = target.integral0(mdm_array[i],target.roi)
                    i1_array[i,:] = target.integral1(mdm_array[i],mZl,target.roi)
                    i2_array[i,:] = target.integral2(mdm_array[i],mZl,target.roi)
                    for j in range(target.nelements):
                        E0_array[i,j,:] = target.integrand0(mdm_array[i],sample,j)
                        E1_array[i,j,:] = target.integrand1(mdm_array[i],mZl,sample,j)
                        E2_array[i,j,:] = target.integrand2(mdm_array[i],mZl,sample,j)
                    asimovres = optimize.minimize(stat.unbinned.llbl,1,args=(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mdm_array[i],mZl,0,gh,order[i]))
                    asimov = asimovres.fun
                    const = quadranthop(stat.unbinned.llglpos,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mdm_array[i],mZl,gh,order[i]),('pe','pe'))
                    dest = quadranthop(stat.unbinned.llglpos,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mdm_array[i],mZl,-gh,order[i]),('pe','pe'))
                    if const.success == False or const.globalfun>=asimov:
                            mlgl[i,0] = 0
                            mlfun[i,0] = asimov
                            mlNsig[i,0]=stat.unbinned.Ntot(target,mdm_array[i],mZl,0,gh,0*ibg,1,i0_array[i,:],i1_array[i,:],i2_array[i,:])
                    else:
                        mlgl[i,0] = const.globalmin[1]
                        mlfun[i,0] = const.globalfun
                        mlNsig[i,0] = stat.unbinned.Ntot(target,mdm_array[i],mZl,mlgl[i,0]*order[i],gh,0*ibg,1,i0_array[i,:],i1_array[i,:],i2_array[i,:])

                    if dest.success == False or dest.globalfun>=asimov:
                        mlgl[i,1] = 0
                        mlfun[i,1] = asimov
                        mlNsig[i,1] = mlNsig[i,0]
                    else:
                        mlgl[i,1] = dest.globalmin[1]
                        mlfun[i,1] = dest.globalfun
                        mlNsig[i,1] = stat.unbinned.Ntot(target,mdm_array[i],mZl,mlgl[i,1]*order[i],-gh,0*ibg,1,i0_array[i,:],i1_array[i,:],i2_array[i,:])
                else:
                    mlgl[i,0] = 0 
                    mlgl[i,1] = 0

            if np.count_nonzero(mlgl[:,0])<0.2*n:
                mlgl[:,0] = np.zeros(n)


            for i in tqdm(range(n)):
                # find exclusion limits
                # constructive
                Emax = target.E_max(mdm_array[i])
                if Emax > (thr-2*res):
                    if mlgl[i,0] == 0:
                        rbic = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,0],mlNsig[i,0],chi,mdm_array[i],mZl,gh,order[i]),
                                        1e-3,1e4,sep=1.001)
                        if rbic.success == True:
                            limgl_h[i,0] = rbic.x0
                            limgl_l[i,0] = np.NaN
                        else:
                            limgl_h[i,0] = np.NaN
                            limgl_l[i,0] = np.NaN
                    else:
                        rbic = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,0],mlNsig[i,0],chi,mdm_array[i],mZl,gh,order[i]),
                                        mlgl[i,0],1e4,sep=1.001)
                        if rbic.success == True:
                            limgl_h[i,0] = rbic.x0
                        else:
                            limgl_h[i,0] = np.NaN
                        limgl_l[i,0] = np.NaN

                    # destructive
                    if mlgl[i,1] == 0:
                        rbid = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,1],mlNsig[i,1],chi,mdm_array[i],mZl,-gh,order[i]),
                                        1e-3,1e4,sep=1.001)
                        if rbid.success == True:
                            limgl_h[i,1] = rbid.x0
                            limgl_l[i,1] = np.NaN
                        else:
                            limgl_h[i,1] = np.NaN
                            limgl_l[i,1] = np.NaN
                    else:
                        rbid = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,1],mlNsig[i,1],chi,mdm_array[i],mZl,-gh,order[i]),
                                        mlgl[i,1],1e4,sep=1.001)
                        lbid = lbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,1],mlNsig[i,1],chi,mdm_array[i],mZl,-gh,order[i]),
                                        1e-3,mlgl[i,1],sep=1.001)
                        if rbic.success == True:
                            limgl_h[i,1] = rbid.x0
                            limgl_l[i,1] = lbid.x0
                        else:
                            limgl_h[i,1] = np.NaN
                            limgl_l[i,1] = np.NaN
                else:
                    limgl_h[i,0] = np.NaN
                    limgl_l[i,0] = np.NaN
                    limgl_h[i,1] = np.NaN
                    limgl_l[i,1] = np.NaN

            outputdict = {  "med. mass" : mZl,
                            "gh" : gh,
                            "gl order" : list(order),
                            "gl ML constructive" : list(mlgl[:,0]),
                            "gl ML destructive" : list(mlgl[:,1]),
                            "gl high constructive" : list(limgl_h[:,0]),
                            "gl low constructive" : list(limgl_l[:,0]),
                            "gl high destructive" : list(limgl_h[:,1]),
                            "gl low destructive" : list(limgl_l[:,1]),
                            }

            split_ = args.sample.split('_')
            targetname = split_[0]
            targetid = split_[-1].split('.')[0]
            if args.id:
                fileid = args.id
                filename = args.plane + "_" + targetname + "_" + targetid + "_" + fileid + ".json"
            else:
                filename = args.plane + "_" + targetname + "_" + targetid + ".json"

            whereto_out = os.path.join(whereto, filename)

            with open(whereto_out, 'w') as fp:
                json.dump(outputdict, fp,  indent=4)
            fp.close

        elif args.plane == "mZgl":
            mZ_array = np.logspace(-1,2,n)
            order = np.zeros(n)
            # in the first column is always the constructive, in the second column
            # the destructive case
            mlgl = np.zeros([n,2])
            mlfun = np.zeros([n,2])
            mlNsig = np.zeros([n,2])
            limgl_h = np.zeros([n,2])
            limgl_l = np.zeros([n,2])
            # first index for mass, second index for target element
            i0_array = np.zeros([n,target.nelements])
            i1_array = np.zeros([n,target.nelements])
            i2_array = np.zeros([n,target.nelements])
            # first index for mass, second index for target element, third for the sample member
            E0_array = np.zeros([n,target.nelements,N_sample])
            E1_array = np.zeros([n,target.nelements,N_sample])
            E2_array = np.zeros([n,target.nelements,N_sample])

            mdm = args.dm
            gh = args.gh

            for i in tqdm(range(n)):
                order[i] = stat.getglorder(N_sample,target,mdm,mZ_array[i])
                i0_array[i,:] = target.integral0(mdm,target.roi)
                i1_array[i,:] = target.integral1(mdm,mZ_array[i],target.roi)
                i2_array[i,:] = target.integral2(mdm,mZ_array[i],target.roi)
                for j in range(target.nelements):
                    E0_array[i,j,:] = target.integrand0(mdm,sample,j)
                    E1_array[i,j,:] = target.integrand1(mdm,mZ_array[i],sample,j)
                    E2_array[i,j,:] = target.integrand2(mdm,mZ_array[i],sample,j)
                const = quadranthop(stat.unbinned.llglpos,(sample,bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mdm,mZ_array[i],gh,order[i]),('pe','pe'))
                dest = quadranthop(stat.unbinned.llglpos,(sample,bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mdm,mZ_array[i],-gh,order[i]),('pe','pe'))
                asimovres = optimize.minimize(stat.unbinned.llbl,1,args=(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mdm,mZ_array[i],0,gh,order[i]))
                asimov = asimovres.fun
                if const.success == False or const.globalfun>=asimov:
                    mlgl[i,0] = 0
                    mlfun[i,0] = asimov
                    mlNsig[i,0]=stat.unbinned.Ntot(target,mdm,mZ_array[i],0,gh,0*ibg,1,i0_array[i,:],i1_array[i,:],i2_array[i,:])
                else:
                    mlgl[i,0] = const.globalmin[1]
                    mlfun[i,0] = const.globalfun
                    mlNsig[i,0] = stat.unbinned.Ntot(target,mdm,mZ_array[i],mlgl[i,0]*order[i],gh,0*ibg,1,i0_array[i,:],i1_array[i,:],i2_array[i,:])

                if dest.success == False or dest.globalfun>=asimov:
                    mlgl[i,1] = 0
                    mlfun[i,1] = asimov
                    mlNsig[i,1] = mlNsig[i,0]
                else:
                    mlgl[i,1] = dest.globalmin[0]
                    mlfun[i,1] = dest.globalfun
                    mlNsig[i,1] = stat.unbinned.Ntot(target,mdm,mZ_array[i],mlgl[i,1]*order[i],-gh,0*ibg,1,i0_array[i,:],i1_array[i,:],i2_array[i,:])

            if np.count_nonzero(mlgl[:,0])<0.2*n:
                mlgl[:,0] = np.zeros(n)


            for i in tqdm(range(n)):
                # find exclusion limits
                # constructive
                if mlgl[i,0] == 0:
                    rbic = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,0],mlNsig[i,0],chi,mdm,mZ_array[i],gh,order[i]),
                                    1e-3,1e4,sep=1.001)
                    if rbic.success == True:
                        limgl_h[i,0] = rbic.x0
                        limgl_l[i,0] = np.NaN
                    else:
                        limgl_h[i,0] = np.NaN
                        limgl_l[i,0] = np.NaN
                else:
                    rbic = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,0],mlNsig[i,0],chi,mdm,mZ_array[i],gh,order[i]),
                                    mlgl[i,0],1e4,sep=1.001)
                    if rbic.success == True:
                        limgl_h[i,0] = rbic.x0
                    else:
                        limgl_h[i,0] = np.NaN
                    limgl_l[i,0] = np.NaN

                # destructive
                if mlgl[i,1] == 0:
                    rbid = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,1],mlNsig[i,1],chi,mdm,mZ_array[i],-gh,order[i]),
                                    1e-3,1e4,sep=1.001)
                    if rbid.success == True:
                        limgl_h[i,1] = rbid.x0
                        limgl_l[i,1] = np.NaN
                    else:
                        limgl_h[i,1] = np.NaN
                        limgl_l[i,1] = np.NaN
                else:
                    rbid = rbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,1],mlNsig[i,1],chi,mdm,mZ_array[i],-gh,order[i]),
                                    mlgl[i,1],1e4,sep=1.001)
                    lbid = lbisection(stat.unbinned.ts_excl_gl,(bg_array,E0_array[i,:,:],E1_array[i,:,:],E2_array[i,:,:],ibg,i0_array[i,:],i1_array[i,:],i2_array[i,:],target,mlfun[i,1],mlNsig[i,1],chi,mdm,mZ_array[i],-gh,order[i]),
                                    1e-4,mlgl[i,1],sep=1.001)
                    if rbic.success == True:
                        limgl_h[i,1] = rbid.x0
                        limgl_l[i,1] = lbid.x0
                    else:
                        limgl_h[i,1] = np.NaN
                        limgl_l[i,1] = np.NaN


            outputdict = {  "dm mass" : mdm,
                            "gh" : gh,
                            "gl order" : list(order),
                            "gl ML constructive" : list(mlgl[:,0]),
                            "gl ML destructive" : list(mlgl[:,1]),
                            "gl high constructive" : list(limgl_h[:,0]),
                            "gl low constructive" : list(limgl_l[:,0]),
                            "gl high destructive" : list(limgl_h[:,1]),
                            "gl low destructive" : list(limgl_l[:,1]),
                            }

            split_ = args.sample.split('_')
            targetname = split_[0]
            targetid = split_[-1].split('.')[0]
            if args.id:
                fileid = args.id
                filename = args.plane + "_" + targetname + "_" + targetid + "_" + fileid + ".json"
            else:
                filename = args.plane + "_" + targetname + "_" + targetid + ".json"



            whereto_out = os.path.join(whereto, filename)

            with open(whereto_out, 'w') as fp:
                json.dump(outputdict, fp,  indent=4)
            fp.close

        else:
            raise NameError("Exclusion plane not yet implemented")