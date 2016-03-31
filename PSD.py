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
import os
from mpl_toolkits.mplot3d import Axes3D
from optparse import OptionParser
import sys
from oneton import pipath
import matplotlib as mpl
if os.name == 'nt':
    import winsound

class analysis():
    '''
    This is a class to handle PSD analysis.
    '''

    def __init__(self, path=None, MaxNumFiles=1000):
        if path is not None:
            self.path = path
        self.MaxNumFiles = MaxNumFiles
        #define some graphics defaults
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['legend.fontsize'] = 16
        mpl.rcParams['axes.titlesize'] = 20
        mpl.rcParams['xtick.labelsize'] = 16
        mpl.rcParams['ytick.labelsize'] = 16
        mpl.rcParams['figure.subplot.bottom'] = 0.12
        mpl.rcParams['figure.subplot.left'] = 0.12
        mpl.rcParams['image.cmap'] = 'Blues'
        if os.name == 'nt':
            self.isWindows = True
        self.FOM = None
        self.PSDinfo = None
        self.MaxURelFOM = 0.1
        self.pltTitle = ''
        self.goodidx = -1
        return

    def ReadFile(self,Filename):
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

    def ReadFileBatch(self,Filename):
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

    def ReadFileFast(self,Filename):
        wfm = []
        with open(Filename, 'r') as f:
            for i in range(5): f.readline()
            #wfm = [float(line.split()[2]) for line in f]
            wfm = [float(line[27:]) for line in f]
            #print wfm[0:10]
        return wfm

    def GetPSD(self, wfm, startoff, fastint, totint):
        minidx = np.argmin(wfm)
        #sum BL for 400 pts prior to trig
        BL = sum(wfm[int(minidx-startoff-400):int(minidx-startoff)])/400
        Qfast = sum(wfm[int(minidx-startoff):int(minidx+fastint)]) - BL*(startoff+fastint)
        Qtot = sum(wfm[int(minidx-startoff):int(minidx+totint)]) - BL*(startoff+totint)
        return (Qfast,Qtot)

    def DoubleGauss(self, x, p):
        #A1, mu1, sigma1, A2, mu2, sigma2 = p
        return p[0]*np.exp(-(x-p[2])**2/(2.*p[3]**2)) + p[1]*np.exp(-(x-p[4])**2/(2.*p[5]**2))
    
    def ErrorFunc(self, p, x, y):
        return self.DoubleGauss(x,p) - y

    #Old method; 1 integration time per call
    #def EvalPSD(mypath, FastInt, LongInt, IntRange, MaxNumFiles,
    #            PSDrange, QtotRange):
    #New method: all integration times per call
    def EvalPSD(self, AllFastTimes, AllLongTimes, IntRange, MaxNumFiles,
                PSDrange, QtotRange):
        #Function to be used to evaluate PSD for optimising LongInt and FastInt
        onlyfiles = [ f for f in listdir(self.path) if isfile(join(self.path,f))&("Data_" in f) ]
        PSDinfo = []
        numfiles = 0
        AvgWfm = np.zeros(10000)
        for fname in onlyfiles:
            if numfiles>MaxNumFiles: break
            if np.remainder(numfiles,2000)==0:
                print(numfiles, " files read!")
                print("The next file is:", fname)
            numfiles+=1
            Wfm = self.ReadFileFast(join(self.path,fname))
            #Fill PSDinfo sequentially using several FastInt and LongInt.
            IntValues = [[x,y] for x in AllFastTimes for y in AllLongTimes]
            PSDinfo.append([self.GetPSD(Wfm,100,Value[0],Value[1]) for Value in IntValues])
        
            #if numfiles==1: continue                
            #thisPSD = PSDinfo[-1][0]/PSDinfo[-1][1]
            #print "thisPSD = ", thisPSD, " Qtot = ", PSDinfo[-1][1]
            #if (PSDrange[0]<thisPSD)&(thisPSD<PSDrange[1])\
            #&(QtotRange[0]<PSDinfo[-1][1])&(PSDinfo[-1][1]<QtotRange[1]):
                #AvgWfm += Wfm
                #print "Passed AvgWfmCuts! fname = ", fname
            
        print('Original PSDinfo shape = ' + str(np.shape(PSDinfo)))
        PSDinfo = np.transpose(PSDinfo, [1,2,0])
        print('Transposed PSDinfo shape = ' + str(np.shape(PSDinfo)))
        #IntValues = np.transpose(IntValues)
        #plt.figure()
        #Arbirarily plot PSD of 5th tpeak and ttail values
        #plt.hist2d(PSDinfo[5][1], PSDinfo[5][0]/PSDinfo[5][1], (100,200), range=((-450,-300), (0.0, 1.0)))
        #Select range from -300:-180 as it appears flat for a guessed PSD value.
        #10% WbLS range = -80:-30
        #For the PROSPECT data, (-450,-300) is about right.
        print("FastInt = ", IntRange[0], "LongInt = ", IntRange[1])
        return [[self.GetFOM(PSDinfo[i], (50, 200), IntRange, (0.0,1.0)) for i in range(len(PSDinfo))],
                IntValues, AvgWfm, PSDinfo]

    def GetFOM(self, PSDinfo, nbins, rangex, rangey):
        FOM = []
        PSDgamma = []
        PSDneut = []
        thehist = np.histogram2d(PSDinfo[1], PSDinfo[0]/PSDinfo[1], nbins, range=(rangex, rangey))
        pkdata = sum(thehist[0][:],0)#sum(np.transpose(thehist[0][:]),1)
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
            print("Failed to find 2 peaks, returning...")
            return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres,\
            coeff, uncerts, FOMuncert

        p0 = [max(pkdata), max(pkdata), bin_centres[min(pkguess)], 0.03, bin_centres[max(pkguess)], 0.03]
        #Use bootstrap method; this neglects covariaces, but the covariances are
        #small between the variables used in FOM calc.    
        coeff, uncerts = self.fit_function(p0, bin_centres, pkdata, self.DoubleGauss)        
        #print("coeff = " + str(coeff))
        #print("uncerts = " + str(uncerts))
        try:    
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
            lastval = (coeff[4]-coeff[2])/(2.35*((coeff[3]+coeff[5])**2))
            Jacobian = [firstval, -firstval, lastval, lastval]#AKA sensitivity coeff matrix
            #When using fit_function, use this:
            #print "Jacobian = ", Jacobian
            FOMuncert = np.sqrt(np.dot(np.multiply(Jacobian,Jacobian),
                                       np.multiply(uncerts[2:],uncerts[2:])))
            #When using curve_fit or leastsqr covariance matrix; use this:    
            #FOMuncert = np.sqrt(np.dot(Jacobian,
            #                           np.dot(uncerts,np.transpose(Jacobian))))
            #print "result = ", result        
            #coeff, var_matrix, infodict = result
        except RuntimeError or IndexError:
            FOM = 0
            PSDgamma = 0
            PSDneut = 0
            var_matrix=[]
            FOMuncert = 0
            print("Could not find a fit, returning...")
            return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres, var_matrix, FOMuncert
        
        #fitted = DoubleGauss(bin_centres, *coeff)
        FOM = abs((abs(coeff[4])-abs(coeff[2]))/(2.35*(abs(coeff[3])+abs(coeff[5]))))
        PSDgamma = max(coeff[2],coeff[4])
        PSDneut = min(coeff[2],coeff[4])
        #print "numfiles = ", numfiles    
    
        print("FOM = ", FOM, " +/- ", FOMuncert)
        print("PSDgamma = ", PSDgamma, ", PSDneut = ", PSDneut)
        return FOM, PSDgamma, PSDneut, PSDinfo, pkdata, bin_centres,\
        coeff, uncerts, FOMuncert


    def fit_function(self, p0, datax, datay, function, **kwargs):
        
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

    def SetMaxNumFiles(self,num):
        self.MaxNumFiles = num
        return

    def PSD(self, AllFastTimes, AllLongTimes, IntRange=(-450,-300)):
        PSDrange=(0,0.8)#for averaging, CURRENTLY UNUSED AND NEEDS FIXING.
        QtotRange=(-350,-250)#EJ-309 = (-450,-350). CURRENTLY UNUSED AND NEEDS FIXING.
        self.IntValues = []
        self.PSDinfo = []
        stuff, IntValues, AvgWfm, self.PSDinfo = \
            self.EvalPSD(AllFastTimes, AllLongTimes, IntRange, self.MaxNumFiles,
                         PSDrange, QtotRange)
        self.FOM, self.PSDgamma, self.PSDneut = \
            np.transpose([stuff[i][0:3] for i in range(len(stuff))])
        self.pkdata = [stuff[i][4] for i in range(len(stuff))]
        self.bin_centres = stuff[0][5]
        self.coeff, self.uncerts, self.FOMuncert = \
            np.transpose([stuff[i][6:9] for i in range(len(stuff))])
        self.IntValues = np.transpose(IntValues)
        return
        
    def getBestIntValues(self):
        '''
        Find the integration times that give the largest FOM.
        
        This needs to be called after the PSD method.
        The relative uncertainty of the biggest FOM must be less than MaxURelFOM,
        to avoid choosing unstable/spurious values.
        returns Qfast, Qtotal, index of best FOM.
        '''
        if self.FOM is None:
            print("PSD.getBestIntValues ERROR: FOM optimisation hasn't been done yet")
            return None
        FOMmax = 0.
        for i,FOM in enumerate(self.FOM):
            if self.FOMuncert[i]/FOM < self.MaxURelFOM and FOM > FOMmax:
                FOMmax = FOM
                self.goodidx = i
        if self.goodidx == -1:
            print("PSD.getBestIntValues: No good FOM found.")
            return None
        return self.IntValues[0][self.goodidx], self.IntValues[1][self.goodidx], self.goodidx

    def pltFOMoptim(self, savefig=True, fname="FOMvsIntegrationTimes.svg",
                    cmap=plt.cm.winter):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(self.IntValues[0], self.IntValues[1], self.FOM, cmap=cmap, linewidth=0.2)
        ax.set_xlabel("Fast Integration Time (ns)", fontsize=16)
        ax.set_ylabel("Total Integration Time (ns)", fontsize=16)
        ax.set_zlabel("Figure of Merit", fontsize=16)
        ax.set_zlim(0,5)
        ax.set_title(self.pltTitle, fontsize=16)
        if savefig:
            plt.savefig(join(self.path,fname))
        return fig, ax

    def pltPSD(self, idx, savefig=True, fname='PSDplot.svg', qmax=100):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist2d(-self.PSDinfo[idx][1], 
                  1 - np.divide(self.PSDinfo[idx][0],self.PSDinfo[idx][1]),
                  (200,200), range=((0,qmax), (0.0, 1.0)))
        ax.set_xlabel("Pulse charge (AU)", fontsize=16)
        ax.set_ylabel("PSD", fontsize=16)
        ax.set_title(self.pltTitle)
        if savefig:
            plt.savefig(join(self.path,fname))
        return fig, ax

    def pltQtot(self, idx, savefig=True, fname='Qtotplot.svg', qmax=100, nbins=1000):
        fig = plt.figure()
        ax = fig.gca()
        ax.hist(-self.PSDinfo[idx][1], bins=nbins, range=(0,qmax))
        ax.set_xlabel("Pulse charge (AU)", fontsize=16)
        ax.set_ylabel("Counts", fontsize=16)
        ax.set_title(self.pltTitle)
        if savefig:
            plt.savefig(join(self.path, fname))
        return fig, ax

    def getFOMvsCharge(self, PSDidx, edgecharge, nbins, qmax, qmin=0):
        #HQEPMT: qmax=80#EJ-309 = 800, DBLS = 500, P20=800
        self.FOMvsCharge = []
        self.UFOMvsCharge = []
        self.pkdatavsCharge = []
        self.coeffvsCharge = []
        binwidth=(qmax-qmin)/(nbins-1)
        ComptonEdge = 477.65 #in keV for Cs137    
        self.EnCal = ComptonEdge/edgecharge #keV/bin
        self.Energies = np.linspace(qmin, qmax, nbins)*self.EnCal
        self.UEnergies = [self.EnCal*0.5*binwidth for i in range(len(self.Energies))]
        for i in np.linspace(qmin, qmax, nbins):
            dummy1, dummy2, dummy3, dummy4, dummy5, dummy6, dummy7, dummy8, dummy9 = \
            anal.GetFOM(anal.PSDinfo[50], (50,300), (-(i+binwidth),-i), (0.,1.))
            #EJ-309_surf/Li bin = 20, EJ-309 bin = , DBLS bin = , P-20 bin = 88
            self.FOMvsCharge.append(dummy1)
            self.pkdatavsCharge.append(dummy5)
            self.coeffvsCharge.append(dummy7)
            self.UFOMvsCharge.append(dummy9)

        plt.figure()
        plt.errorbar(self.Energies, self.FOMvsCharge, xerr=self.UEnergies, yerr=self.UFOMvsCharge)
        plt.xlabel('Electron Equivalent Energy (keV)')
        plt.ylabel('PSD FOM')
        plt.title(self.pltTitle)
        return

if __name__ == '__main__' :
    parser = OptionParser()
    (options,args) = parser.parse_args(args=sys.argv)
    if len(args)>1:
        mypath = pipath.pipath.fix(args[1])
    else:
        #assume it's the current dir
        mypath = os.getcwd()
        print("No path was provided. Using the current directory: {0}".format(mypath))
        
    anal = analysis(path=mypath, MaxNumFiles=30000)
    #EJ-309 data
    #mypath = '/home/lbignell/PSD Analysis/EJ-309/AmBeAndCs137/3522338134/'
    #DBLS data
    #mypath = '/home/lbignell/PSD Analysis/DBLS/Cs137AndAmBe150817/'
    #mypath = os.path.normpath("C:/Users/lbignell/Desktop/CFN Proposal Jun 2015/PSD Measurements/Data/DBLS/AmBeAndCs137150817/")
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
    #mypath = os.path.normpath("C:/Users/lbignell/Desktop/CFN Proposal Jun 2015/PSD Measurements/Data/PROSPECT/EJ-309/AmBeAndCs137_151014/3527782480/")
    #EJ-309 + surfactant
    #mypath = '/home/lbignell/PSD Analysis/PROSPECT/EJ-309_Surf/AmBeAndCs137_151019/'
    #mypath = os.path.normpath("C:/Users/lbignell/Desktop/CFN Proposal Jun 2015/PSD Measurements/Data/PROSPECT/EJ-309_Surf/CollimatedAmBeAndCs137_151029/3528977124/")
    #EJ-309 + surfactant + Li
    #mypath = '/home/lbignell/PSD Analysis/PROSPECT/EJ-309_Surf_Li/AmBeAndCs137_151016/'
    #mypath = os.path.normpath("C:/Users/lbignell/Desktop/CFN Proposal Jun 2015/PSD Measurements/Data/PROSPECT/EJ-309_Surf_Li/CollimatedAmBeAndCs137_151027/3528908578")
    #P-20
    #mypath = os.path.normpath("C:/Users/lbignell/Desktop/PSD Measurements/Data/PROSPECT/P-20/AmBeAndCs137_151110/3530197699")
    

    #Do a nelder-mead optimisation on the LongInt and FastInt parameters
    #results = minimize(EvalPSD,)
    #Actually, those optimisations find it difficult to stick to integer values.
    #I'll try a grid search instead. I already know the 'best' values are
    #approximately (fast, short) = (50, 500), so I'll search around there...
    t = time.time()
    AllFastTimes = np.linspace(10, 120, 12)
    AllLongTimes = np.linspace(200, 1000, 9)
    BestFastTime = 0
    BestLongTime = 0
    theidx = -1
    #Select range from -300:-180 as it appears flat for a guessed PSD value.
    #10% WbLS range = -80:-30
    #For the PROSPECT data, (-450,-300) is about right.
    IntRange = (-40,-30)#HQE PMT#(-450, -300)#(-350,-250)#EJ-309 = (-450,-350), DBLS = (-350, -250)
    anal.pltTitle = input('Give the title for the plots: \n')
    anal.PSD(AllFastTimes, AllLongTimes, IntRange)
    ret = anal.getBestIntValues()
    if ret is not None:
        BestFastTime, BestLongTime, theidx = ret
        print("Run Finished! Optimal value of (Fast, Total) integration is (", \
        BestFastTime, ",", BestLongTime, "), for a maximum FOM of ", anal.FOM[theidx])
        elapsed = time.time()-t
        print("Time elapsed = ", elapsed, " seconds")
        anal.pltFOMoptim()
        qmax = max(-anal.PSDinfo[theidx][1])
        anal.pltPSD(theidx, qmax=qmax)
        anal.pltQtot(theidx, qmax=qmax)
        if anal.isWindows:
            winsound.Beep(4000,100)

        EdgeCharge = float(input('Please enter Compton edge location (charge bin): \n'))
        anal.getFOMvsCharge(theidx, EdgeCharge, 20, qmax=qmax, qmin=0)
    else:
        print('Couldn\'t find a good PSD, returning...')
