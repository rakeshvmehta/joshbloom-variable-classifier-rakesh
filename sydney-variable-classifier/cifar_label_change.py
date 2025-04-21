import random 
import pickle
import numpy as np
import argparse

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def class_dict(txt):
    sub2class = {}
    with open(txt) as fl:
        for line in fl:
            line=line.split(":")
            clss = line[0].replace(" ","_")
            subclss = line[1].split(",")
           # d = dict([(x,0) for x in a])
            sub2class = {**sub2class, **dict([(c.strip().replace(" ","_"),clss) for c in subclss])}
    return sub2class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change some cifar 100 labels to parent labels')

    parser.add_argument('--frequency', default=.2,
                        help="What fraction (between 0 and 1) of labels to convert to parent labels"
                        )
    parser.add_argument('--cifar_fl', default='cifar/cifar-100-python/train',
                        help="Cifar data file"
                        )
    parser.add_argument('--txt', default='cifar/cifar-100-python/class_subclass.txt',
                        help="text file containing names of classes, one group of classes per line "
                        ) 
    
    args = parser.parse_args()

    sub2class = class_dict(args.txt) 

    child_ind = list(range(0,100))
    child_label = sorted(list(sub2class.keys()))
    child_label2ind = dict([(child_label[i],child_ind[i]) for i in range(len(child_ind))])
    child_ind2label = dict([(child_ind[i],child_label[i]) for i in range(len(child_ind))])

    parent_ind = list(range(100,100+len(np.unique(list(sub2class.values())))))
    parent_label = np.unique(list(sub2class.values()))
    parent_label2ind = dict([(parent_label[i],parent_ind[i]) for i in range(len(parent_ind))])
    parent_ind2label = dict([(parent_ind[i],parent_label[i]) for i in range(len(parent_ind))])

    sub2class_ind = dict([(child_label2ind[key], parent_label2ind[sub2class[key]]) for key in sub2class.keys()])

    data = unpickle(args.cifar_fl)
    labels = data[b'fine_labels']
    labels_final = [l if random.random() > args.frequency else sub2class_ind[l] for l in labels]

    imgs = data[b'data']

    pickle.dump({'data':imgs, 'labels':labels_final, 'child_ind2label':child_ind2label, 'parent_ind2label':parent_ind2label}, open( "cifar/cifar-100-python/cifar_data.pkl", "wb" ) )