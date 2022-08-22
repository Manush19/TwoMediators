# -*- coding: utf-8 -*-
"""
@author: einfaltleonie
"""

import numpy as np
from numpy import linalg
from scipy import optimize
import iminuit as imin
import operator

class HoppingResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    globalmin : ndarray
        The global minimum found.
    globalfun : ndarray
        Values of objective function at the global minimum. 
    minima : ndarray
        All minima found.
    count : list
        Counts of how often each minimum was found.
    fun : ndarray
        Values of objective function at all found minima.
    success : bool
        Whether or not the optimizer exited successfully.
    message : str
        Description of the cause of the termination.
    precision: str
        States if there was a termination due to precision loss in 
        the routines, where the result was still accepted as a success.
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

def zerothstep(function, func_args, guess0, varkey, negative):
    foundmin = False
    count = 0
    output = {}
    message = ''
    x0 = 0
    fun0 = 0
    while foundmin == False:
        fun0 = function(guess0,*list(func_args))
        if np.isfinite(fun0) == True:
            mini0 = optimize.minimize(function,guess0,args=func_args) #imin.minimize(function, guess0, args=func_args) 
            if mini0.success == True or (all(mini0.x!=guess0) and np.isfinite(mini0.x).all() == True):
                fun0 = mini0.fun
                x0 = mini0.x
                foundmin = True
                break
            else:
                message = 'Can not evaluate at zeroth guess'   
        guess0 = np.zeros(1)
        if np.size(varkey) == 1:
            if varkey[1] == "e":
                ng = np.random.exponential(0.5)
            elif varkey[1] == "u":
                ng = np.random.uniform(1,10)
            guess0 = np.concatenate((guess0,[ng]))
        else:
            for var in varkey:
                if var[1] == "e":
                    ng = np.random.exponential(0.5)
                elif var[1] == "u":
                    ng = np.random.uniform(1,10)
                guess0 = np.concatenate((guess0,[ng]))
        guess0 = guess0[1:]
        count +=1
        if count>30:
            message = 'Can not find good zeroth guess, enter manually'
            break
    output = {'guess0': guess0, 'fun0': fun0, 'x0': x0}
    errorhandler = {'error': operator.not_(foundmin) , 'message': message}
    return errorhandler, output


def firstquadstep(function, func_args, varkey, fun0):
    stepsuccess = False
    count = 0
    while stepsuccess == False:
        nextguess = np.zeros(1)
        if np.size(varkey) == 1:
            if varkey[1] == "e":
                ng = np.random.exponential(0.5)
            elif varkey[1] == "u":
                ng = np.random.uniform(1,10)
            nextguess = np.concatenate((nextguess,[ng]))
        else:
            for var in varkey:
                if var[1] == "e":
                    ng = np.random.exponential(0.5)
                elif var[1] == "u":
                    ng = np.random.uniform(1,10)
                nextguess = np.concatenate((nextguess,[ng]))
        nextguess = nextguess[1:]
        xguess = function(nextguess,*func_args)
        if xguess < fun0*5 and np.isfinite(xguess) == True:
            stepsuccess = True
            break
        else:
            count += 1
        if count == 30:
            break
    return nextguess

def secondquadstep(function, func_args, varkey, fun0):
    stepsuccess = False
    count = 0
    while stepsuccess == False:
        nextguess = np.zeros(1)
        if np.size(varkey) == 1:
            if varkey[1] == "e":
                ng = np.random.exponential(0.5)
            elif varkey[1] == "u":
                ng = np.random.uniform(1,10)
            if varkey[0] == "n":
                ng = -ng
            nextguess = np.concatenate((nextguess,[ng]))
        else:
            for var in varkey:
                if var[1] == "e":
                    ng = np.random.exponential(0.5)
                elif var[1] == "u":
                    ng = np.random.uniform(1,10)
                if var[0] == "n":
                    ng = -ng
                nextguess = np.concatenate((nextguess,[ng]))
        nextguess = nextguess[1:]        
        xguess = function(nextguess,*func_args)
        if xguess < fun0*5 and np.isfinite(xguess) == True:
            stepsuccess = True
            break
        else:
            count += 1
        if count == 30:
            break
    return nextguess

def xiknown(result, x, fun, count):
    xi = np.array([result.x])
    prev = np.where(np.all(abs(x-xi)<0.1, axis=1))[0]
    if prev.size:
        count[prev[0]] += 1
    else:
        x = np.concatenate((x, xi), axis = 0)
        fun.append(result.fun)
        count.append(1)
    return x, fun, count

def firstquadsearch(function, func_args, varkey, failiter, multicount, guess0, 
                    x0, fun0, terminationflag, precisionflag):
    #short function to test if the xi is an already known one
    termflag = terminationflag
    precflag = precisionflag   
    count = []  # count of how often a certain minima was found
    fun = []    # list of all according function values
    fun.append(fun0), count.append(1)
    x = np.array([x0])
    count_fail = 0
    while count_fail < failiter:
        guess = firstquadstep(function, func_args, varkey, fun0)
        result = optimize.minimize(function,guess,args=func_args)
        if result.success == True:
            x, fun, count = xiknown(result, x, fun, count)
        elif result.success == False and all(result.x!=guess) and np.isfinite(result.x).all() == True: 
            x, fun, count = xiknown(result, x, fun, count)
            precflag = True 
        else:
            count_fail += 1
        if max(count) >= multicount:
            break
        if count_fail > failiter:
            termflag = True
            break
    return { 'x': x, 'fun': fun, 'count':count, 
            'termination': termflag, 'precision': precflag, 'nfail': count_fail}

def secondquadsearch(function, func_args, varkey, failiter, multicount,
                     x, fun, count, terminationflag, precisionflag):
    termflag = terminationflag
    precflag = precisionflag
    count_fail = 0
    while count_fail < failiter:
        guess = secondquadstep(function, func_args, varkey, fun[0])
        result = optimize.minimize(function,guess,args=func_args)
        if result.success == True:
            x, fun, count = xiknown(result, x, fun, count)
        elif result.success == False and all(result.x!=guess) and np.isfinite(result.x).all() == True: 
            x, fun, count = xiknown(result, x, fun, count)
            precflag = True 
        else:
            count_fail += 1
        if max(count) >= multicount*2:
            break
        if count_fail > failiter:
            termflag = True
            break
    return {'x': x, 'fun': fun, 'count':count, 
            'termination': termflag, 'precision': precflag, 'nfail': count_fail}

def quadranthop(function, func_args, varkey, **kwargs):
    """ A hopping algorithm to find a global minimum in the presence of 
        various local minima and undefined functional values for wide paramter ranges.
        No guess/starting value is needed, it will be drawn from an exponential or uniform distribution.
    Attributes
    ----------
    function : callable
        The objective function which should be minimized.
    func_args : tuple
        Extra arguments passed to the objective function.
    varkey : tuple of strings
        One key for each free parameter which should be minimized. First character either 'p' for 
        only positive guesses or 'n' for negative and positive guesses. Only one of the free parameters
        can have allowed negative values (only 'two quadrants' allowed). Second character to specifiy
        if the guess should be drawn from an uniform distribution 'u' from 1 to 10 or from an 
        exponential distribution 'e' with t=0.5.
    guess0 : float (optional)
        If you want to give a guess value anyways.
    failiter : integer (optional)
        Number of allowed failed iterations before the routine is stopped, default is 5.
    multicount: integer (optional)
        Number of times one minimum needs to be found before the rountine stops searching. If
        varkey is 'p' default is 4, for varkey 'n' default is 2 (since for n we go through two search
        routines so 2x2=4).

    """
    # read out kwargs
    if 'guess' in kwargs:
        guess0 = kwargs.get('guess')
    else:
        guess0 = np.zeros(1)
        if np.size(varkey) == 1:
            if varkey[1] == "e":
                ng = 0.1
            elif varkey[1] == "u":
                ng = 1
            guess0 = np.concatenate((guess0,[ng]))
        else:
            for var in varkey:
                if var[1] == "e":
                    ng = 0.1
                elif var[1] == "u":
                    ng = 1
                guess0 = np.concatenate((guess0,[ng]))
        guess0 = guess0[1:]
    if 'failiter' in kwargs:
        failiter = kwargs.get('failiter')
    else:
        failiter = 5 
    # check if there are negative values to be handled:
    check = 'ne','nu'
    if any(c in varkey for c in check):
        negative = True
    else:
        negative = False
    if 'multicount' in kwargs:
        multicount = kwargs.get('multicount')
    elif negative == True:
        multicount = 2
    else: 
        multicount = 4
    success = False
    x = 0
    fun = 0
    globalmin = x
    globalfun = fun
    count = 0
    precision = 'Not applicable.'
    # create error flags
    error = False
    precisionflag = False
    terminationflag = False
    errorhandler, stepzero = zerothstep(function, func_args, guess0, varkey, negative)
    error = errorhandler.get('error')
    if error == False:
        x0 = stepzero.get('x0')
        fun0 = stepzero.get('fun0')
        #searching the first quadrant
        quad = firstquadsearch(function, func_args, varkey, failiter, multicount, guess0, 
                    x0, fun0, terminationflag, precisionflag)
        precisionflag = quad['precision']
        terminationflag = quad['termination']
        #do we need to search in a second quadrant?
        if negative == True:
            #searching the second quadrant
            quad2 = secondquadsearch(function, func_args, varkey, failiter, multicount,
                    quad['x'], quad['fun'],
                    quad['count'], terminationflag, precisionflag) 
            quad = quad2       
        #extracting the results and checking for overlaps in the two
        #quadrant searches
        x = quad['x']
        count = quad['count']
        fun = quad['fun']
        precisionflag = quad.get('precision')
        terminationflag = quad.get('termination')
        globalfun = min(fun)
        globalmin = x[fun.index(globalfun),:]
        #some message and error control:
        if terminationflag is True:
            success = False
            message = 'Termination due to too many failed minimization routines.'
        else:
            success = True
            message = 'At least one minima was found a sufficient number of times.'
        if precisionflag is True:
            precision = 'At some point a termination due to precision loss occured'
        else:
            precision = 'No precision loss in any of the minimizations'
    else:
        message = errorhandler.get('message')

    result = HoppingResult(success = success, precision = precision, message = message,
                           globalfun = globalfun, fun = fun, minima = x, globalmin = globalmin,
                           count = count)
    return result 

