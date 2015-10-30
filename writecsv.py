# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 15:24:33 2015

@author: lbignell
"""
#only takes 1D list and writes it to a csv.
def writecsv(filepath, data):
    import csv
    with open(filepath, 'wb') as csvfile:
        thewriter = csv.writer(csvfile)
        thewriter.writerow(data)
        
Path = '/home/lbignell/PSD Analysis/PROSPECT/AvgWfm_EJ309_surf_Li.csv'
Data = AvgWfm_EJ309_surf_Li
writecsv(Path, Data)