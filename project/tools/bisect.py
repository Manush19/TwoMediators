# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np
from numpy import linalg
from scipy import optimize

class BisectResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    x0 : float
        The maximum root found.
    it : int
        Number of iterations needed.
    success : bool
        Whether or not the bisection exited successfully.
    message : str
        Description of the cause of the termination.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def rbisection(function, func_args, minx, maxx, **kwargs):
    """ The bisection algorithm to find the value where a 
        strictly monotonously increasing function stops being zero. 
    
    Attributes
    ----------
    function : callable
        The objective function for which x0 should be found.
    func_args : tuple
        Extra arguments passed to the objective function.
    minx : float
        Minimum value of searched interval.
    maxx : float
        Maximum value of searched interval.
    sep : float, optional
        Maximum seperation between the found root and the next bisection value, for
        the bisection to exit successfully: x1<=sep*x0 (1.001 by default)
    maxiter : float, optional
        The maximum number of bisection steps (default: 50).

    """
    # read out kwargs
    if 'sep' in kwargs:
        sep = kwargs.get('sep')
    else:
        sep = 1.001
    if 'maxiter' in kwargs:
        maxiter = kwargs.get('maxiter')
    else:
        maxiter = 50

    count = 0
    success = False
    x0 = minx
    x2 = maxx
    while count < maxiter:
        x1 = np.sqrt(x0*x2) #(x2-x0)/2+x0
        f1 = function(x1,*func_args)
        if f1 == 0:
            if x1 <= sep*x0:
                success = True
                break
            else:
                x0 = x1
        else:
            x2 = x1
        count += 1

    result = BisectResult(success = success,  x0 = x0, it = count)
    return result 

def lbisection(function, func_args, minx, maxx, **kwargs):
    """ The bisection algorithm to find the value where a 
        strictly monotonously falling function starts to be zero. 
    
    Attributes
    ----------
    function : callable
        The objective function for which x0 should be found.
    func_args : tuple
        Extra arguments passed to the objective function.
    minx : float
        Minimum value of searched interval.
    maxx : float
        Maximum value of searched interval.
    sep : float, optional
        Maximum seperation between the found root and the next bisection value, for
        the bisection to exit successfully: x1<=sep*x0 (1.001 by default)
    maxiter : float, optional
        The maximum number of bisection steps (default: 50).

    """
    # read out kwargs
    if 'sep' in kwargs:
        sep = kwargs.get('sep')
    else:
        sep = 1.001
    if 'maxiter' in kwargs:
        maxiter = kwargs.get('maxiter')
    else:
        maxiter = 50

    count = 0
    success = False
    x0 = minx
    x2 = maxx
    #print (minx,maxx,'lbisection')
    while count < maxiter:
        x1 = (x0+x2)/2.#np.sqrt(x0*x2) #(x2-x0)/2+x0 #  #
        f1 = function(x1,*func_args)
       # print (x1,f1)
        if f1 == 0:
            if x1 <= sep*x0:
                success = True
                break
            else:
                x2 = x1
        else:
            x0 = x1
        count += 1

    result = BisectResult(success = success, x0 = x0,  it = count)
    return result 
