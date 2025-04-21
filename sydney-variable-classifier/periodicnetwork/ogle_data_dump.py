import astropy.table as table
import numpy as np
import joblib
import argparse
import os
from light_curve import LightCurve

#data not included on github

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert full star tables and photometry data to form compatible with code')

    parser.add_argument('--min_obs', default=300,
                        help="Minimum number of observations per object"
                        )
    parser.add_argument('--outname', default="ogle_data.pkl",
                        help="Output filename"
                        )
    
    args = parser.parse_args()
    
    types = ['acep','cep','dpv','lpv','rcb','rrlyr','sct','t2cep']
    lightcurves = []
    for t in types:
        data = np.genfromtxt(t+'.txt',names=True, dtype=None, skip_header=6, usecols=range(37), encoding=None)
        data = table.Table(data) #Actually taking first 20,000 objects -- only relevant for LPV and RRLyr, should randomize in future
        for row in data: 
            pathname = 'photometry/'+t+'/I/'+row['ID']+'.dat'
            if os.path.isfile(pathname):
                lc = np.loadtxt(pathname,skiprows=0)
                if len(lc[:,0]) >= args.min_obs:
                    label = row['Type'] + '_' + row['Subtype'] if (row['Subtype']!=-99.99 and row['Subtype']!='-99.99') else row['Type']
                    lightcurves.append(LightCurve(lc[:,0],lc[:,1],lc[:,2],label=label,p=row['P_1'])) #I think these are the periods (from Fourier decomposition?)?
    joblib.dump(lightcurves, args.outname)
    
    labels = [lc.label for lc in lightcurves]
    unique, counts = np.unique(labels, return_counts=True)
    labels_dict = {unique[i]: counts[i] for i in range(len(unique))} 
    print(labels_dict)