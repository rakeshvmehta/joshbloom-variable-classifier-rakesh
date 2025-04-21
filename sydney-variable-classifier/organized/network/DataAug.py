import argparse
import os, glob
import gzip, tarfile
import joblib
import urllib.request
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from light_curve import LightCurve

from astropy.stats import sigma_clip
#!pip install gatspy
#from gatspy.periodic import LombScargle

from tqdm import tqdm_notebook

#Modified version of functions from https://github.com/jorgemarpa/PELS-VAE/blob/master/notebooks/preprocess_OGLE3.ipynb
def preprocess_lc(lc, sig=3, ite=3):
    # sigma clipping in phot outlier
    clip_mag = sigma_clip(lc[:,1], sigma=sig,
                          maxiters=ite, cenfunc=np.median,
                          copy=False)
    # sigma clipping in err values
    clip_err = sigma_clip(lc[:,2], sigma=sig,
                          maxiters=ite, cenfunc=np.median,
                          copy=False)
    lc = lc[(~clip_mag.mask) & (~clip_err.mask)]
    return lc

def subsample(lc, nb=200):
    # undersample light curve to nb observations
    # if len(lc) > nb, undersample 
    # else, concatenate true lc with extra observations
    # sampled from N(m, m_err)
    if len(lc.measurements) >= nb:
        new_lc = copy_lc(lc)#lc.copy()
        ind = np.random.choice(len(lc.measurements), size=nb, replace=False)
        new_lc.measurements = lc.measurements[ind]
        new_lc.times = lc.times[ind]
        new_lc.errors = lc.errors[ind]
        return new_lc
    else:
        new_lc = copy_lc(lc)
        ind = np.random.choice(len(lc.measurements), size=nb-len(lc.measurements), replace=True)
        ind = np.asarray(ind).astype(int)
        new_lc.measurements = lc.measurements[ind]
        new_lc.times = lc.times[ind]
        new_lc.errors = lc.errors[ind]
        return np.vstack([lc, extra])

def sample_err(lc):
    # resample observations from N(m, m_err)
    new_lc = copy_lc(lc)#lc.copy()
    new_lc.measurements = np.random.normal(loc=lc.measurements, scale=lc.errors)
    return new_lc
    
#My functions
def copy_lc(lc):
  return LightCurve(lc.times, lc.measurements, lc.errors, survey=lc.survey, name=lc.name, best_period=lc.best_period, best_score=lc.best_score, label=lc.label, p=lc.p, p_signif=lc.p_signif, p_class=lc.p_class, ss_resid=lc.ss_resid)
  
def give_me_lc(filename='ogle_data.pkl', force=True):
    data = joblib.load('data/{}'.format(filename))
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find embedding or similarity of class(es)')

    parser.add_argument('--file', default='ogle_data.pkl',
                        help="Input file")
    parser.add_argument('--outname', default='data/ogle_data_aug.pkl',
                        help="Name of output pickle file")
    parser.add_argument('--max', default=100,
                        help="Maximum samples per class")
    parser.add_argument('--min', default=60,
                        help="Minimum samples per class")
    parser.add_argument('--threshold', default=40,
                        help="Threshold for augmenting to min or max")

    data = give_me_lc(args.file)
    unique_label, count = np.unique([lc.label for lc in data], return_counts=True)
    target = [args.min if c < args.threshold else args.max for c in count]
    label2count = {unique_label[i]:count[i] for i in range(len(count))}
    label2target = {unique_label[i]:target[i] for i in range(len(target))}
    
    NB = 200
    Plot = True
    aug_lcs = []
    for label in unique_label: #k, (vt, nn) in enumerate(meta_new.Type.value_counts().sort_index().items()):
        count = label2count[label]
        target = label2target[label]
        class_data = [lc for lc in data if lc.label == label]
        aa = 0
        print(count,target)
        while count < target:
            rnd_idx = np.random.choice(len(class_data))
            org_lc = class_data[rnd_idx]
            P = org_lc.p
            new_lc = subsample(phase_shift(sample_err(org_lc),P), nb=NB)
            aug_lcs.append(new_lc)
            count += 1
        if count > target: 
            rnd_idxs = np.random.choice(len(class_data), target, replace=False)
            aug_lcs.extend([class_data[i] for i in rnd_idxs])
        else:
            aug_lcs.extend(class_data)
                   
    new_data = aug_lcs
    unique_label, count = np.unique([lc.label for lc in new_data], return_counts=True)
    label2count = {unique_label[i]:count[i] for i in range(len(count))}
    print(label2count)

    data_name = args.file.split("_")[0]
    output = 'data/Aug_data/' + data_name + '_aug/' + data_name + '_' + 'max' + str(args.max) + '_min' + str(args.min) + '.pkl'
    joblib.dump(new_data, args.outname)