import argparse, sys, json, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
sys.path.append('../../../')
from project.constants import atomic
import project.tools.statistics as stat
from project.constants import cosmo

# class SmartFormatter(argparse.HelpFormatter):

#     def _split_lines(self, text, width):
#         if text.startswith('R|'):
#             return text[2:].splitlines()  
#         # this is the RawTextHelpFormatter._split_lines
#         return argparse.HelpFormatter._split_lines(self, text, width)

# targethelp = ("R|Specificing the material of the target.\n"
#             +"Input element name followed directly by the\n"
#             +"multiplicity of the element.\n"
#             +"As an example: NaI --> Na1 I1, Al2O3 --> Al2 O3")
# modelhelp = ("R|Specifications of the DM-SM interaction model with\n" 
#             +"two mediators. Creates a list and the input has\n"
#             +"to be in the following order: \n"
#             +"DM mass in GeV; \n"
#             +"light mediator mass in MeV; \n"
#             +"light mediator coupling; \n"
#             +"effective heavy mediator coupling in 1/MeV^2; \n"
#             +"'c' or 'd', for constructive or destructive case")
# bghelp = ("R|Description of the background in (keV kg day)^-1\n"
#             +"Single value input: \n"
#             +"flat background in (default: 0)\n"
#             +"2 value input: \n"
#             +"slope and intercept of linear background")
# funchelp = ("This program reads in parameters of an experiment (target material, threshold, exposure, ..) in -t and of "
#             +"a two mediator or single mediator model in -m (just set one coupling to 0). It then generates and saves "
#             +"a mocksample to a json file.")

        
# parser = argparse.ArgumentParser(description=funchelp, formatter_class=SmartFormatter)
# parser.add_argument("-ps", "--plotspectrum", action="store_true", help="Plots a differential recoil spectrum for the given data, logarithmic y scale.")
# parser.add_argument("-ph", "--plothisto", action="store_true", help="Plots a histogram of the generated mocksample.")
# parser.add_argument("-t", "--target", nargs='+', required=True, help=targethelp)
# parser.add_argument("-m", "--model", default=(1,1,0,0,'c'), nargs=5, help=modelhelp)
# parser.add_argument("-ex", default=1000, type=float, help="Exposure in kg days (default: 1000)")
# parser.add_argument("-id", help="identification key for the sample, can be number or string")
# parser.add_argument("-th", default=1e-4, type=float, help="Threshold in keV (default: 0)")
# parser.add_argument("-re", default=0, type=float, help="Sigma for Gaussian resolution in keV (default: 0)")
# parser.add_argument("-bg", default=[0], nargs='+', type=float, help=bghelp)
# parser.add_argument("-roi", type=float, help="Upper limit of ROI, needs to be given for 0 or flat background")
# args = parser.parse_args()

def samples(id,model,targets = ['Na1', 'I1'], exposure = 50,thr = 1,res = 0.2,bg = [1],roi = 40,plot_types = ['ph','ps'],seed = None, **kwargs):
    """ For making mock samples
    -----------
    id: identification key for the sample, can be number or string
    model: Specifications of the DM-SM interaction model with two mediators.
           to be in the order [DM mass, light mediator mass, light mediator -
           coupling, effective heavy mediator coupling, 'c' for construcitve or 
           'd' for destructive]
    targets: Specifing the material of the target. Input element name followed 
             directly by the multiplicity of the element. default ['Na1', 'I1'] 
    exposure: Exposure in kg days (default = 50 kg days)
    thr: Threshold in keV (default = 1 keV)
    res: Sigma for Gaussian resolution in keV (default = 0.2 keV)
    bg: Description of the background in (keV kg day)^-1.  (defualt = [1])
        Single value input for flat background.
        2 value input, slope and intercept for linear background.
    roi: Upper limit of ROI, needs to be given for flat background. (defaul = 40 keV)
    plot_types: list of plots to make, 'ps' for ploting differential recoilspecturm,
                'ph' for plotitng a histogram of the generated mock sample.
    seed: seed for intializing the random number generator used to create the mock sample.
          default is None, for which mock data created will have different poisson 
          statistics each time this function is called.
    **kwargs: Kwargs for specifying the VDF informations necessary. Only to be passed if 
              the default VDF of SHM with vesc 544 km/s and vcir 220 km/s has to be changed.
              For VDF initializations (see initialize_vdf in tools.statistics.__core.py)
              'vdf','ve': The 1D vdf in eath ref frame corresponding to 've' velocity in 
              earth's reference frame.
              'vcir': circular velocity in km/s of the local standard of rest (default = 220)
              'vesc': local escape velocity in km/s (default = 544 km/s)
    """
    class Arguments:
        def __init__(self,targets, model, exposure, id, thr, res, bg, roi, plot_types):
            self.target = targets
            self.model = model
            self.ex = exposure
            self.id = id
            self.th = thr
            self.re = res
            self.bg = bg
            self.roi = roi
            if 'ps' in plot_types:
                self.plotspectrum = True
            else:
                self.plotspectrum = False
            if 'ph' in plot_types:
                self.plothisto = True
            else:
                self.plothisto = False
            
    
    # check for output directory:
    pathflag = False
    whereto = os.path.join(os.getcwd(), "../Output")
    if not os.path.isdir(whereto):
        print ("no output directory found")
        return None
    if os.access(whereto, os.R_OK):
        pathflag = True
    else:
        print ("readable_dir:{0} is not a readable dir".format(whereto))
        return None

    args = Arguments(targets, model, exposure, id, thr, res, bg, roi, plot_types)
    
    # define the target class:
    material = []
    name = ""
    for t in args.target:
        m = [atomic.A[t[0:-1]],float(t[-1])]
        material.append(m)
        name += t[0:-1] 
        if float(t[-1])>1:
            name += str(t[-1])

    if args.roi:
        roimax = args.roi
        target = stat.target(*material, exposure = args.ex, thr = args.th,
                            sigma = args.re, bg = args.bg, roi = args.roi)
    else: 
        target = stat.target(*material, exposure = args.ex, thr = args.th,
                            sigma = args.re, bg = args.bg)
        roimax = target.roimax
        
    if 'vdf' in kwargs:
        vdf = kwargs.get('vdf')
        I_index = 2
        ve = kwargs.get('ve')
    else:
        I_index = 0
        vdf = ['SHM']
        ve = ['SHM']
        
    if 'vesc' in kwargs:
        vesc = kwargs.get('vesc')
        if I_index == 0:
            I_index = 1
    else:
        vesc = cosmo.v_esc
        
    if 'vcir' in kwargs:
        vcir = kwargs.get('vcir')
        if I_index == 0:
            I_index = 1
    else:
        vcir = 220.0
        
    if I_index == 2:
        target.initialize_vdf(I_index = 2, VDF = vdf, V = ve, vesc = vesc, vcir = vcir)
    elif I_index == 1:
        target.initialize_vdf(I_index = 1, vesc = vesc, vcir = vcir)
    else:
        target.initialize_vdf()

    # define model-dependent paramenters
    m_dm = float(args.model[0])
    m_Z_l= float(args.model[1])
    g_l = float(args.model[2])
    if args.model[4] == 'c':
        g_heff = np.abs(float(args.model[3]))
    else:
        g_heff = -np.abs(float(args.model[3]))


    E = np.linspace(args.th,roimax,int(1e3))
    #N = target.Ntotsimp(m_dm,m_Z_l,g_l,g_heff)
    labelplot = name #+ "\n N in ROI: {:.2f}".format(N)


    if args.plotspectrum:
        if pathflag == True:
            if args.id:
                plotname = name + args.id + "_spectrum" + ".png"
            else:
                plotname = name + "_spectrum" + ".png"
            wheretoplot = os.path.join(whereto, plotname)
            fig1 = plt.figure('differential event rate',
                            constrained_layout=False, figsize=(6,4))
            ax = fig1.add_subplot()
            ax.set_xlabel(r'$E_r\,\,[keV]$')
            ax.set_ylabel(r'$dR/dE_r\,\,[(keV\,kg\,day)^{-1}]$')
            ax.tick_params(which='both',direction='in')
            ax.set_yscale('log')
            ax.set_xlim(args.th,roimax)
            #ax.set_ylim(1e-80,1e-10)
            ax.plot(E,target.diffrecoil(m_dm,m_Z_l,g_l,g_heff,E), color = 'teal', linewidth=1, label = labelplot)
            ax.legend()
            fig1.savefig(wheretoplot, dpi=400)
            plt.clf()


    if g_l == 0 and g_heff == 0:
        #here we generate the mocksample without signal
        sample = stat.bgsample(target,seed)
    else:
        #here we generate the mocksample with signal
        sample = stat.mocksample(target,m_dm,m_Z_l,g_l,g_heff,seed)
    output = {
        "target" : {
            "material" : material,
            "exposure" : args.ex,
            "thr" : args.th,
            "sigma" : args.re,
            "bg" : args.bg,
            "roi" : roimax
        },
        "model" : {
            "mdm" : m_dm,
            "mZl" : m_Z_l,
            "gl" : g_l,
            "gheff" : g_heff
        },
        "sample" : list(sample["sample"]),
        'vdf': list(vdf),
        've': list(ve),
        'vesc':vesc,
        'vcir':vcir,
        'I_index':I_index
    }
    labelhisto = name + " N=" + str(np.size(sample['sample']))
    """
    siz = 0
    for sam in sample['sample']:
        if sam > args.th:
            siz+=1
    labelhisto = name + " N=" + str(siz)
    """
    if args.plothisto:
        if pathflag == True:
            if args.id:
                histoname = name + args.id + "_histogram" + ".png"
            else:
                histoname = name + "_histogram" + ".png"
            wheretohisto = os.path.join(whereto, histoname)
            fig2 = plt.figure('mock sample histogram',
                    constrained_layout=False, figsize=(6,4))
            ax = fig2.add_subplot()
            ax.set_xlabel(r'$E_r\,\,[keV]$')
            ax.tick_params(which='both',direction='in')
            ax.hist(sample["sample"],10,label=labelhisto, histtype='step')#color = 'teal', alpha = 0.5)
            #ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.set_xlim(0,roimax)
            ax.legend()
            fig2.savefig(wheretohisto, dpi=400)
            plt.clf()

    if args.id:
        name = name +"_" + args.id + "_mock"  + ".json"
    else:
        name = name + "_mock"  + ".json"
    wheretofile = os.path.join(whereto, name)
    with open(wheretofile, 'w') as fp:
        json.dump(output, fp,  indent=4)
    fp.close
    print('Saved sample data to {} in the ouput folder.'.format(name))

