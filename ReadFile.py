# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:32:44 2015

@author: lbignell
"""
import time
from os import listdir
from os.path import isfile, join
import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import peakutils
from scipy.optimize import minimize

def ReadFile(Filename):
    wfm = []
    with open(Filename, 'r') as f:
        f.next()
        f.next()
        f.next()
        f.next()
        f.next()
        for line in f:
            #print(line.split())            
            thestr = line.split()
            #print(type(thestr[2]))
            theval = float(thestr[2])
            wfm.append(theval)
           
    return wfm

def ReadFileBatch(Filename):
    f = open(Filename)
    lines = f.readlines(5)
    chunksize = 100000
    lines = f.readlines(chunksize)
    wfm = []
    while lines:
        for line in lines:
            #print(line)
            thestr = line.split()
            theval = float(thestr[2])
            wfm.append(theval)
            
        lines = f.readlines(chunksize)
            
    return wfm

def ReadFileFast(Filename):
    wfm = []
    with open(Filename, 'r') as f:
        f.next()
        f.next()
        f.next()
        f.next()
        f.next()
        #wfm = [float(line.split()[2]) for line in f]
        wfm = [float(line[27:]) for line in f]
        #print wfm[0:10]
    return wfm

def GetPSD(wfm, startoff, fastint, totint):
    minidx = np.argmin(wfm)
    #sum BL for 400 pts prior to trig
    BL = sum(wfm[int(minidx-startoff-400):int(minidx-startoff)])/400
    Qfast = sum(wfm[int(minidx-startoff):int(minidx+fastint)]) - BL*(startoff+fastint)
    Qtot = sum(wfm[int(minidx-startoff):int(minidx+totint)]) - BL*(startoff+totint)
    return (Qfast,Qtot)
    
def DoubleGauss(x, p):
    #A1, mu1, sigma1, A2, mu2, sigma2 = p
    #return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2)) + A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
    #print "p = ", p    
    return p[0]*np.exp(-(x-p[2])**2/(2.*p[3]**2)) + p[1]*np.exp(-(x-p[4])**2/(2.*p[5]**2))
    
def ErrorFunc(p, x, y):
    #print "ErrorFunc! x = ", x, " y = ", y, " p = ", p
    return DoubleGauss(x,p) - y

def EvalPSD(mypath, FastInt, LongInt, IntRange, MaxNumFiles,
            PSDrange, QtotRange):
    #Function to be used to evaluate PSD for optimising LongInt and FastInt
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))&("Data_" in f) ]
    #for thisFastInt in FastInt:
    PSDinfo = []
    numfiles = 0
    #maxnumfiles=5000.0
    AvgWfm = np.zeros(10000)
    for fname in onlyfiles:
        #print "fname = ", fname
        if numfiles>MaxNumFiles: break
        if np.remainder(numfiles,2000)==0: print numfiles, " files read!"
        #if numfiles>3400: print "filename = ", fname
        numfiles+=1
        Wfm = ReadFileFast(join(mypath,fname))
        #PSDinfo.append(GetPSD(Wfm,100,thisFastInt,LongInt))
        PSDinfo.append(GetPSD(Wfm,100,FastInt,LongInt))
        if numfiles==1: continue                
        thisPSD = PSDinfo[-1][0]/PSDinfo[-1][1]
        #print "thisPSD = ", thisPSD, " Qtot = ", PSDinfo[-1][1]
        if (PSDrange[0]<thisPSD)&(thisPSD<PSDrange[1])\
        &(QtotRange[0]<PSDinfo[-1][1])&(PSDinfo[-1][1]<QtotRange[1]):
            AvgWfm += Wfm
            #print "Passed AvgWfmCuts! fname = ", fname
            

    PSDinfo = np.transpose(PSDinfo)    
    #plt.figure()
    #plt.hist2d(PSDinfo[1], PSDinfo[0]/PSDinfo[1], (100,200), range=((-50,-0), (0.0, 1.0)))
    #Select range from -300:-180 as it appears flat for a guessed PSD value.
    #10% WbLS range = -80:-30
    #For the PROSPECT data, (-450,-300) is about right.
    print "FastInt = ", IntRange[0], "LongInt = ", IntRange[1]
    return (GetFOM(PSDinfo, (50, 200), IntRange, (0.0,1.0)), AvgWfm)

def GetFOM(PSDinfo, nbins, rangex, rangey):
    FOM = []
    PSDgamma = []
    PSDneut = []
    thehist = np.histogram2d(PSDinfo[1], PSDinfo[0]/PSDinfo[1], nbins, range=(rangex, rangey))
    pkdata = sum(np.transpose(thehist[0][:]),1)
    bin_centres = (thehist[2][:-1] + thehist[2][1:])/2
    #Crude estimate of peak location
    #plt.figure()
    #plt.plot(bin_centres, pkdata)
    pkguess = peakutils.peak.indexes(pkdata, 0.1, 3)#thresh was 0.2
    if np.size(pkguess)<1:
        FOM = 0
        PSDgamma = 0
        PSDneut = 0
        coeff = []
        uncerts = []
        FOMuncert=0
        print "Failed to find 2 peaks, returning..."
        return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres,\
        coeff, uncerts, FOMuncert

    #print "pkguess = ", pkguess, " size = ", np.size(pkguess)
    p0 = [100, 100, bin_centres[min(pkguess)], 0.03, bin_centres[max(pkguess)], 0.03]
    #Use bootstrap method; this neglects covariaces, but the covariances are
    #small between the variables used in FOM calc.    
    coeff, uncerts = fit_function(p0, bin_centres, pkdata, DoubleGauss)        
    #print "coeff = ", coeff
    #print "uncerts = ", uncerts    
    #try:    
        #coeff, var_matrix = curve_fit(DoubleGauss, bin_centres, sum(np.transpose(thehist[0][:]), 1), p0)
        #coeff, var_matrix, infodict, mesg, ier =\
        #leastsq(ErrorFunc, p0,
        #        args=(bin_centres, sum(np.transpose(thehist[0][:]), 1)),
        #        full_output=True)
        #print "ier = ", ier
        #print "mesg = ", mesg
        #print "size(var_matrix) = ", size(var_matrix)
        #if ier>4 or size(var_matrix)==1:
        #if np.size(var_matrix)==1:
        #    FOM = 0
        #    PSDgamma = 0
        #    PSDneut = 0
        #    var_matrix=[]
        #    FOMuncert = 0
        #    print "Could not find a fit, returning..."
        #    return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres, var_matrix, FOMuncert
            
        #s_sqr = (infodict['fvec']**2).sum()/(np.size(bin_centres)-6)
        #var_matrix = var_matrix*s_sqr    
        #var_matrix = var_matrix[2::, 2::]
    firstval = (1/(2.35*(coeff[3]+coeff[5])))
    lastval = (coeff[4]-coeff[2])/(2.35*(coeff[3]+coeff[5]))
    Jacobian = [firstval, -firstval, lastval, lastval]
    #When using fit_function, use this:
    #print "Jacobian = ", Jacobian
    FOMuncert = np.sqrt(np.dot(np.multiply(Jacobian,Jacobian),
                        np.multiply(uncerts[2:],uncerts[2:])))
    #When using curve_fit or leastsqr covariance matrix; use this:    
    #FOMuncert = np.sqrt(np.dot(Jacobian,
    #                           np.dot(uncerts,np.transpose(Jacobian))))
        #print "result = ", result        
        #coeff, var_matrix, infodict = result
    #except RuntimeError:
        #FOM = 0
        #PSDgamma = 0
        #PSDneut = 0
        #var_matrix=[]
        #FOMuncert = 0
        #print "Could not find a fit, returning..."
        #return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres, var_matrix, FOMuncert
        
    #fitted = DoubleGauss(bin_centres, *coeff)
    FOM = abs((coeff[4]-coeff[2])/(2.35*(coeff[3]+coeff[5])))
    PSDgamma = max(coeff[2],coeff[4])
    PSDneut = min(coeff[2],coeff[4])
    #print "numfiles = ", numfiles    
    
    print "FOM = ", FOM, " +/- ", FOMuncert
    print "PSDgamma = ", PSDgamma, ", PSDneut = ", PSDneut
    return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres,\
    coeff, uncerts, FOMuncert


def fit_function(p0, datax, datay, function, **kwargs):

    errfunc = lambda p, x, y: function(x,p) - y
    #print "p0 = ", p0

    ##################################################
    ## 1. COMPUTE THE FIT AND FIT ERRORS USING leastsq
    ##################################################

    # If using optimize.leastsq, the covariance returned is the 
    # reduced covariance or fractional covariance, as explained
    # here :
    # http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es
    # One can multiply it by the reduced chi squared, s_sq, as 
    # it is done in the more recenly implemented scipy.curve_fit
    # The errors in the parameters are then the square root of the 
    # diagonal elements.   
    pfit, pcov, infodict, errmsg, success = \
        leastsq( errfunc, p0, args=(datax, datay), \
                          full_output=1)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append( np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    #print "cov matrix = ", pcov


    ###################################################
    ## 2. COMPUTE THE FIT AND FIT ERRORS USING curvefit
    ###################################################

    # When you have an error associated with each dataY point you can use 
    # scipy.curve_fit to give relative weights in the least-squares problem. 
    datayerrors = kwargs.get('datayerrors', None)
    #curve_fit_function = kwargs.get('curve_fit_function', function)
    #if datayerrors is None:
    #    pfit, pcov = \
    #        curve_fit(curve_fit_function,datax,datay,p0=p0)
    #else:
    #    pfit, pcov = \
    #         curve_fit(curve_fit_function,datax,datay,p0=p0,\
    #                            sigma=datayerrors)
    #error = [] 
    #for i in range(len(pfit)):
    #    try:
    #      error.append( np.absolute(pcov[i][i])**0.5)
    #    except:
    #      error.append( 0.00 )
    #pfit_curvefit = pfit
    #perr_curvefit = np.array(error)  


    ####################################################
    ## 3. COMPUTE THE FIT AND FIT ERRORS USING bootstrap
    ####################################################        

    # An issue arises with scipy.curve_fit when errors in the y data points
    # are given.  Only the relative errors are used as weights, so the fit
    # parameter errors, determined from the covariance do not depended on the
    # magnitude of the errors in the individual data points.  This is clearly wrong. 
    # 
    # To circumvent this problem I have implemented a simple bootstraping 
    # routine that uses some Monte-Carlo to determine the errors in the fit
    # parameters.  This routines generates random datay points starting from
    # the given datay plus a random variation. 
    #
    # The random variation is determined from average standard deviation of y
    # points in the case where no errors in the y data points are avaiable.
    #
    # If errors in the y data points are available, then the random variation 
    # in each point is determined from its given error. 
    # 
    # A large number of random data sets are produced, each one of the is fitted
    # an in the end the variance of the large number of fit results is used as 
    # the error for the fit parameters. 

    # Estimate the confidence interval of the fitted parameter using
    # the bootstrap Monte-Carlo method
    # http://phe.rockefeller.edu/LogletLab/whitepaper/node17.html
    residuals = errfunc( pfit, datax, datay)
    s_res = np.std(residuals)
    ps = []
    # 100 random data sets are generated and fitted
    for i in range(100):
      if datayerrors is None:
          randomDelta = np.random.normal(0., s_res, len(datay))
          randomdataY = datay + randomDelta
      else:
          randomDelta =  np.array( [ \
                             np.random.normal(0., derr,1)[0] \
                             for derr in datayerrors ] ) 
          randomdataY = datay + randomDelta
      randomfit, randomcov = \
          leastsq( errfunc, p0, args=(datax, randomdataY),\
                            full_output=0)
      ps.append( randomfit ) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit


    # Print results 
    #print "\nlestsq method :"
    #print "pfit = ", pfit_leastsq
    #print "perr = ", perr_leastsq
    #print "\ncurvefit method :"
    #print "pfit = ", pfit_curvefit
    #print "perr = ", perr_curvefit
    #print "\nbootstrap method :"
    #print "pfit = ", pfit_bootstrap
    #print "perr = ", perr_bootstrap
    return pfit_bootstrap, perr_bootstrap

#EJ-309 data
#mypath = '/home/lbignell/PSD Analysis/EJ-309/AmBeAndCs137/3522338134/'
#DBLS data
#mypath = '/home/lbignell/PSD Analysis/DBLS/Cs137AndAmBe150817/'
#10% WbLS data
##Finer range settings
#mypath = '/home/lbignell/PSD Analysis/Sample5/AmBeAndCs137_150821/3523466748_NoLineOfSight/'
##Wider range settings
#mypath = '/home/lbignell/PSD Analysis/Sample5/AmBeAndCs137_150925/'
#1% WbLS data
#mypath = '/home/lbignell/PSD Analysis/Sample1/AmBeAndCs137_150908/'
#PC, 3g/L PPO, 15mg/L bis-MSB
#mypath = '/home/lbignell/PSD Analysis/PC_PPO_BisMSB/AmBeAndCs137_151001/3526571654/'
#PC, 3g/L PPO, 15mg/L bis-MSB
#mypath = '/home/lbignell/PSD Analysis/5pcLAS_3pcPC_3glPPO_15mgLBisMSB/AmBeAndCs137_151002/'
#PROSPECT
#EJ-309
#mypath = '/home/lbignell/PSD Analysis/PROSPECT/EJ-309/AmBeAndCs137_151014/'
#EJ-309 + surfactant
#mypath = '/home/lbignell/PSD Analysis/PROSPECT/EJ-309_Surf/AmBeAndCs137_151019/'
#EJ-309 + surfactant + Li
mypath = '/home/lbignell/PSD Analysis/PROSPECT/EJ-309_Surf_Li/AmBeAndCs137_151016/'


#Do a nelder-mead optimisation on the LongInt and FastInt parameters
#results = minimize(EvalPSD,)
#Actually, those optimisations find it difficult to stick to integer values.
#I'll try a grid search instead. I already know the 'best' values are
#approximately (fast, short) = (50, 500), so I'll search around there...


#AllFastTimes = np.linspace(60, 120, 13)
#AllLongTimes = np.linspace(300, 600, 7)
FastTime = 70
LongTime = 800
#Select range from -300:-180 as it appears flat for a guessed PSD value.
#10% WbLS range = -80:-30
#For the PROSPECT data, (-450,-300) is about right.
IntRange = (-450,-350)
MaxNumFiles = 20000
PSDrange=(0,0.8)
QtotRange=(-450,-350)
#AllFOM = []
#AllPSDgamma = []
#AllPSDneut = []
#Params = []
#MaxFOM = 0
#BestFastTime = 0
#BestLongTime = 0
#for FastTime in AllFastTimes:
#    for LongTime in AllLongTimes:
#stuff, AvgWfm = \
#EvalPSD(mypath, FastTime, LongTime, IntRange, MaxNumFiles,
#        PSDrange, QtotRange)
#FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres, coeff, uncerts, FOMuncert\
#= stuff
#        AllFOM.append(FOM)
#        AllPSDgamma.append(PSDgamma)
#        AllPSDneut.append(PSDneut)
#        Params.append([FastTime,LongTime])
#        if FOM>MaxFOM:
#            MaxFOM = FOM
#            BestFastTime = FastTime
#            BestLongTime = LongTime
FOMvsCharge = []
UFOMvsCharge = []
qmax=2e-8
qmin=0
nbins=51
binwidth=(qmax-qmin)/(nbins-1)
for i in np.linspace(qmin, qmax, nbins):
    dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8, dummy9 = \
    GetFOM(PSDinfo, (50,200), (-(i+binwidth),-i), (0.,1.))
    FOMvsCharge.append(dummy1)
    UFOMvsCharge.append(dummy9)




#print "Run Finished! Optimal value of (Fast, Total) integration is (", \
#BestFastTime, ",", BestLongTime, "), for a maximum FOM of ", MaxFOM

#onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f))&("Data_" in f) ]
#LongInt = 1000
#LongInt = np.linspace(500, 2500, 5)
#FastInt = np.linspace(5,50,10)0.781717563872 , PSDneut =  0.769784452695
#FastInt = 50
#FOM = []
#PSDgamma = []
#PSDneut = []
#for thisFastInt in FastInt:
#for thisLongInt in LongInt:
#    start_time = time.time()
#    PSDinfo = []
#    for fname in onlyfiles:
#        Wfm = ReadFileFast(join(mypath,fname))
        #PSDinfo.append(GetPSD(Wfm,100,thisFastInt,LongInt))
#        PSDinfo.append(GetPSD(Wfm,100,FastInt,thisLongInt))

#    PSDinfo = np.transpose(PSDinfo)    0.781717563872 , PSDneut =  0.769784452695
    #Select range from -300:-180 as it appears flat for a guessed PSD value.
#    thehist = np.histogram2d(PSDinfo[1], PSDinfo[0]/PSDinfo[1], (50,200), range=((-300,-180), (0.0, 1.0)))
#    pkdata = sum(np.transpose(thehist[0][:]),1)
#    bin_centres = (thehist[2][:-1] + thehist[2][1:])/2
    #Crude estimate of peak location
#    pkguess = peakutils.peak.indexes(pkdata, 0.5, 5)
#    p0 = [100, bin_centres[min(pkguess)], 0.03, 100, bin_centres[max(pkguess)], 0.03]
#    coeff, var_matrix = curve_fit(DoubleGauss, bin_centres, sum(np.transpose(thehist[0][:]), 1), p0)
#    fitted = DoubleGauss(bin_centres, *coeff)
    #plt.figure()
    #plt.plot(bin_centres,sum(np.transpose(thehist[0][:]),1),label='data')
    #plt.plot(bin_centres,fitted,label='fit')
#    FOM.append((coeff[4]-coeff[1])/(2.35*(coeff[2]+coeff[5])))
#    PSDgamma.append(max(coeff[1],coeff[4]))
#    PSDneut.append(min(coeff[1],coeff[4]))0.781717563872 , PSDneut =  0.769784452695
#    print("FOM = ", FOM[-1])
#    print("PSDgamma = ", PSDgamma, ", PSDneut = ", PSDneut)
#    print("--- %s seconds ---" % (time.time() - start_time))
    
#plt.figure()
#plt.plot(FastInt, FOM)
#plt.figure()
#plt.plot(LongInt,PSDgamma,label='Gamma PSD')
#plt.plot(LongInt,PSDneut,label='Neutron PSD')
#plt.legend()