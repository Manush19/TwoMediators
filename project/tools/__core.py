# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

"""
This is a private module which is only ever called over init. To
access any of the function below, please do so by calling 'tools.'
"""

import numpy as np
import os
import math
from matplotlib import cm


#color schemes:
def color(map, number, start, stop):
    """Defines an array of colors following a certain color map.
    ----------
    map : string
        Name of color map, see matplotlib documentation.
    number : integer
        Number of colors within the color scheme array. 
    start/stop : float
        Values from 0 to 1, definig from where to where in the full color map one picks the colors.
    """
    cm_subsection = np.linspace(start,stop,number) 
    cm_map = cm.get_cmap(map)
    return [ cm_map(x) for x in cm_subsection ]

def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible 
    given the current xlim of the axis. ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the 
    upper and lower ylims"""

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)

def find_limits(ax,exposure,margin=0.1):
    """This function rescales the x- and y-axis based on the data that is visible 
    given the current exposure. It is however espially defined for 
    expoentially falling recoil spectra!"""
    
    bottom = exposure**(-1)

    def get_top(line):
        yd = line.get_ydata()
        top = np.max(yd)+margin*(np.max(yd)-bottom)
        return top

    def get_right(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = bottom,get_top(line)
        x_displayed = xd[((yd>lo) & (yd<hi))]
        if x_displayed.size:
            h = np.max(x_displayed) - np.min(x_displayed)
            right = np.max(x_displayed)+margin*h
        else:
            right = 0
        return right

    lines = ax.get_lines()
    right = 0
    top = 0

    for line in lines:
        new_top = get_top(line)
        new_right = get_right(line)
        if new_right > right: right = new_right
        if new_top > top: top = new_top
        #if new_right > top: top = new_top

    ax.set_ylim(bottom, top)
    ax.set_xlim(0,right)

def as_si(x, ndp):
    """Formating string ouput in a scientific notation, x is the float and ndp gives the
    decimal places."""
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))

def output_directory(cat,name,*subcat):
    """ Creates a path to the 'output' directory in which the file is saved.
    Attributes
    ----------
    cat : string
        Category of the output, i.e. subfolder, for example 'files' or 'plots'.
    name : string
        Name and ending of the file. 
    subcat : string (optional)
        Subcategory of output, i.e. folder in cat, for example 'slides'
    """
    if subcat:
        directory = os.path.join(os.getcwd(), "tests", "output", cat, subcat[0], name)
    else:
        directory = os.path.join(os.getcwd(), "tests", "output", cat, name)

    return directory

def magnitude(number):
    """Returns the magnitude of a float (number)."""
    return 10**(math.floor(math.log(number, 10)))