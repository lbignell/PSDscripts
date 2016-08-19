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
import matplotlib.pyplot as plt

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
        anal.getFOMvsCharge(theidx, EdgeBin, 35, qmax=qmax, qmin=0)
    if os.name == 'nt':
        winsound.Beep(4000,100)

    return anal
    
def edgerepeatability(basepath, nbins=1000, qmax=100, normed=False):
    '''
    Run the Compton Edge analysis for repeated measurements, and plot.
    Return edge locations.
    Note the only .h5 files in the basepath should be the data files.
    '''
    filelist = os.listdir(basepath)
    Edges = {}
    fig = plt.figure()
    ax = fig.gca()
    for file in filelist:
        if '.h5' in file:
            thisanal = hdfcomptonfile(basepath+file, nbins=nbins)
            ax.hist(-thisanal.PSDinfo[0][1], bins=nbins, range=(0,qmax),
                    normed=normed, histtype='step', label=file)
            Edges[file] = thisanal.EdgeCharge
    return Edges
        

def hdfcomptonfile(path, fasttime=100, longtime=700, intrange=(-40,-30), 
                   makefig=False, nbins=1000):
    '''
    This function will process a Compton edge file, and return the analysis
    object.
    
    fasttime/longtime are in ticks, which are usually 0.2 ns apart.
    '''
    anal = PSDscripts.PSD.analysis(path=path, ishdf=True)
    anal.setComments(anal.wfms.attrs['Comments'])
    anal.pltTitle = anal.wfms.attrs['SampleName']
    anal.PSD([fasttime], [longtime], intrange)
    edgeloc = anal.find_edge_bin(idx=0, makeplot=True, nbins=nbins)
    print('\nFile: {0}\nCompton Edge Bin: {1}\n'.format(path, edgeloc))
    return anal
        
def hdffile(path, fasttimes=np.linspace(10, 120, 12),
             longtimes=np.linspace(200, 1000, 9), intrange=(-40,-30),
             EdgeBin=None, savefigs=True, titplt=None, verbose=1):
    '''
    Do PSD analysis of HDF5 data file. Return an analysis object.
    
    Arguments:
    - path to the data file.
    - fast integration times to scan (ticks)
    - total integration times to scan (ticks)
    - charge range over which to calculate PSD FOM.
    - charge bin location of the Compton Edge.
    - Optionally save FOM optimisation, PSD, charge spectrum, and FOMvsEnergy plots.
    - Verbosity.
    '''
    anal = PSDscripts.PSD.analysis(path=path, ishdf=True)
    anal.setComments(anal.wfms.attrs['Comments'])
    t = time.time()
    BestFastTime = 0
    BestLongTime = 0
    if savefigs and titplt is None:
        anal.pltTitle = anal.wfms.attrs['SampleName']
    anal.PSD(fasttimes, longtimes, intrange)
    ret = anal.getBestIntValues()
    if ret is not None:
        BestFastTime, BestLongTime, theidx = ret
    else:
        vals = [[x,y] for x in fasttimes for y in longtimes]
        print('Couldn\'t find a good PSD, estimating fast,long int as {0}...'
                .format(vals[86]))
        BestFastTime, BestLongTime = vals[86]
        theidx = 86
    print("Run Finished! Optimal value of (Fast, Total) integration is (", \
    BestFastTime, ",", BestLongTime, "), for a maximum FOM of ", anal.FOM[theidx])
    elapsed = time.time()-t
    print("Time elapsed = ", elapsed, " seconds")
    anal.pltFOMoptim(savefig=savefigs)
    qmax = max(-anal.PSDinfo[theidx][1])
    anal.pltPSD(theidx, qmax=qmax, savefig=savefigs)
    anal.pltQtot(theidx, qmax=qmax, savefig=savefigs)
    anal.getFOMvsCharge(theidx, EdgeBin, 35, qmax=qmax, qmin=0, savefig=True)
    if os.name == 'nt':
        winsound.Beep(4000,100)
    return anal
