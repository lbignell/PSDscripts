# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 16:42:29 2016

@author: lbignell
"""
import PSDscripts.PSD
import time
import numpy as np
import os
if os.name == 'nt':
    import winsound

def txtfiles(path, nfiles, fasttimes=np.linspace(10, 120, 12),
             longtimes=np.linspace(200, 1000, 9), intrange=(-40,-30),
             EdgeBin=None, savefigs=True, titplt=None, verbose=1,
             comments=None):
    '''
    Do PSD analysis of nfiles in path. Return an analysis object.
    
    Arguments:
    - path where the text files are
    - maximum number of files to analyse
    - fast integration times to scan (ticks)
    - total integration times to scan (ticks)
    - charge range over which to calculate PSD FOM.
    - charge bin location of the Compton Edge.
    - Optionally save FOM optimisation, PSD, charge spectrum, and FOMvsEnergy plots.
    '''
    anal = PSDscripts.PSD.analysis(path=path, MaxNumFiles=nfiles)
    if comments is not None:
        anal.setComments(comments)
    t = time.time()
    BestFastTime = 0
    BestLongTime = 0
    if savefigs and titplt is None:
        anal.pltTitle = input('Give the title for the plots: \n')
    anal.PSD(fasttimes, longtimes, intrange)
    ret = anal.getBestIntValues()
    if ret is not None:
        BestFastTime, BestLongTime, theidx = ret
    else:
        print('Couldn\'t find a good PSD, returning...')
        return anal
    print("Run Finished! Optimal value of (Fast, Total) integration is (", \
    BestFastTime, ",", BestLongTime, "), for a maximum FOM of ", anal.FOM[theidx])
    elapsed = time.time()-t
    print("Time elapsed = ", elapsed, " seconds")
    anal.pltFOMoptim(savefig=savefigs)
    qmax = max(-anal.PSDinfo[theidx][1])
    anal.pltPSD(theidx, qmax=qmax, savefig=savefigs)
    anal.pltQtot(theidx, qmax=qmax, savefig=savefigs)
    if EdgeBin is not None:
        anal.getFOMvsCharge(theidx, EdgeBin, 20, qmax=qmax, qmin=0)
    if os.name == 'nt':
        winsound.Beep(4000,100)

    return anal