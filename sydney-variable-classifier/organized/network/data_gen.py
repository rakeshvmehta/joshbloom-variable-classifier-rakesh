import torch
import joblib
import numpy as np
import random
import math
from torchvision import datasets, transforms
from util import train_test_split
from DataAug import *

def augment(dataset, size, max_size):
  unique_label, count = np.unique([lc.label for lc in dataset], return_counts=True)
  target = [size for c in count] 
  label2count = {unique_label[i]:count[i] for i in range(len(count))}  # in case add min as well as max size
  label2target = {unique_label[i]:target[i] for i in range(len(target))}
                
  NB = 200
  aug_lcs = []
  for label in unique_label: #k, (vt, nn) in enumerate(meta_new.Type.value_counts().sort_index().items()):
    count = label2count[label]
    target = label2target[label]
    class_data = [lc for lc in dataset if lc.label == label]
    while count < target:
      rnd_idx = np.random.choice(len(class_data))
      org_lc = class_data[rnd_idx]
      P = org_lc.p
      new_lc = subsample(sample_err(org_lc), nb=NB)
      aug_lcs.append(new_lc)
      count += 1
    if count > target: 
        rnd_idxs = np.random.choice(len(class_data), min(count,max_size), replace=False)
        aug_lcs.extend([class_data[i] for i in rnd_idxs])
    else:
        aug_lcs.extend(class_data)
  return aug_lcs
  
def data_check(data):
    # sanity check on dataset
    for lc in data:
        positive = lc.errors > 0
        positive *= lc.errors < 99
        lc.times = lc.times[positive]
        lc.measurements = lc.measurements[positive]
        lc.errors = lc.errors[positive]
        #lc.label = lc.label.replace(" ","")
        lc.p = abs(float(lc.p))
    return data
    
def data_generator(batch_size, dataset, train_size, k=None):
    data = joblib.load(os.path.join("data",dataset))
    data = data_check(data)
    labels = [lc.label for lc in data]
    unique_label, count = np.unique(labels, return_counts=True) 
    data = [lc for lc in data if len(lc.measurements) >= 200]

    if k != None:
        k_lists = [[] for i in range(k)]
        for label in unique_label: 
            class_data = [lc for lc in data if lc.label==label]
            class_labels = [lc.label for lc in class_data]
            random.Random(4).shuffle(class_data)
            class_size = len(class_data)
            
            list_of_lists = [class_data[i:(i+math.ceil(class_size/k))] for i in range(0, class_size, math.ceil(class_size/k))]
            for i in range(k):
                k_lists[i].extend(list_of_lists[i])
        train_sets, val_sets, test_sets = [],[],[]
        for i in range(k): 
            train_indices = [(i+x)%k for x in range(k-2)] 
            val_index = (i+k-2)%k
            test_index = (i+k-1)%k
            
            train_set = np.concatenate([k_lists[i] for i in train_indices])
            #print(train_set)
            val_set = k_lists[val_index]
            test_set = k_lists[test_index]

            new_train_set, new_val_set, new_test_set = [], [], []
            for label in unique_label: 
              class_data = [lc for lc in train_set if lc.label==label]
              if len(class_data) > train_size:
                new_train_set.extend(class_data[:train_size])
                new_val_set.extend(class_data[train_size:train_size+(len(class_data)-train_size)//2])
                new_test_set.extend(class_data[train_size+(len(class_data)-train_size)//2:])
              else:
                new_train_set.extend(class_data)

            val_set.extend(new_val_set)
            test_set.extend(new_test_set)

            datasets = [new_train_set, val_set, test_set]
            sizes = [train_size, 200, 200]
            max_sizes = [train_size, 1000, 1000]

            for d in range(len(datasets)):
                dataset = datasets[d]
                aug_lcs = augment(dataset,sizes[d], max_sizes[d])
                random.Random(4).shuffle(aug_lcs)
                datasets[d] = aug_lcs

            train_sets.append(datasets[0]) 
            val_sets.append(datasets[1])  
            test_sets.append(datasets[2]) 
            
            #train_loader = [torch.utils.data.DataLoader(train_set, batch_size=batch_size) for train_set in train_sets]
            #val_loader = [torch.utils.data.DataLoader(val_set, batch_size=batch_size) for val_set in val_sets]
            #test_loader = [torch.utils.data.DataLoader(test_set, batch_size=batch_size) for test_set in test_sets]
        return train_sets, val_sets, test_sets
    else: 
        train_frac = .8
        labels = np.asarray(labels)
        trains, tests = train_test_split(labels, train_frac,random_state=2)
        train_labels = labels[trains]
        train_frac = .75
        trains, vals = train_test_split(train_labels, train_frac)
        
        data = np.asarray(data)
        test_set = data[tests]
        train_set = data[trains]
        val_set = data[vals]
            
        datasets = [train_set, val_set, test_set]
        min_sizes = [train_size, 200, 200]
        max_sizes = [train_size, 1000, 1000]
        for d in range(len(datasets)):
            dataset = datasets[d]
            aug_lcs = augment(dataset,min_sizes[d], max_sizes[d])
            random.Random(4).shuffle(aug_lcs)
            datasets[d] = aug_lcs
            
        train_loader = torch.utils.data.DataLoader(datasets[0], batch_size=batch_size)
        val_loader = torch.utils.data.DataLoader(datasets[1], batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(datasets[2], batch_size=batch_size)
        
        return datasets
