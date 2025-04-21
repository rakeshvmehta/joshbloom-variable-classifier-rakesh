#Modified version of https://github.com/kmzzhang/periodicnetwork
import os
import sys

sys.path.append('/home/sydney8jenkins/hyperop3/lib64/python3.6/site-packages')
sys.path.append('/home/sydney8jenkins/hyperop3/lib/python3.6/site-packages')
sys.path.append('/home/sydney8jenkins/hyperop3/lib64/python3.6/site-packages/IPython/extensions')
sys.path.append('/home/sydney8jenkins/.ipython')
sys.path.append('/home/sydney8jenkins/hyperop3/lib64/python3.6/site-packages/gitdb/ext/smmap')
print("PATH:",sys.path) 

import joblib
import argparse 
import torch.multiprocessing as mp
import pickle
sys.path.append('./')

parser = argparse.ArgumentParser(description='')  
parser.add_argument('--L', type=int, default=200,#128,
                    help='training sequence length')
parser.add_argument('--filename', type=str, default='ogle_data.pkl',
                    help='dataset filename. file is expected in ./data/')
parser.add_argument('--frac-train', type=float, default=0.8,
                    help='training sequence length')
parser.add_argument('--frac-valid', type=float, default=0.25,
                    help='training sequence length')
parser.add_argument('--train-batch', type=int, default=32,
                    help='training sequence length')
parser.add_argument('--varlen_train', action='store_true', default=False,
                    help='enable variable length training')
parser.add_argument('--input', type=str, default='dtf',
                    help='input representation of data. combination of t/dt/f/df/g.')
parser.add_argument('--n_test', type=int, default=1,
                    help='number of different sequence length to test')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout rate')
parser.add_argument('--dropout-classifier', type=float, default=0,
                    help='dropout rate')
parser.add_argument('--permute', action='store_true', default=False,
                    help='data augmentation')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clipping')
parser.add_argument('--path', type=str, default='results/wandb',#'temp',
                    help='folder name to save experiement results')
parser.add_argument('--max_epoch', type=int, default=20,#50,
                    help='maximum number of training epochs')
parser.add_argument('--min_maxpool', type=int, default=2,
                    help='minimum length required for maxpool operation.')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of gpu devices to use. neg number refer to particular single device number')
parser.add_argument('--njob', type=int, default=1,
                    help='maximum number of networks to train on each gpu')
parser.add_argument('--K', type=int, default=1,#8,
                    help='number of data partition to use')
parser.add_argument('--pseed', type=int, default=0,
                    help='random seed for data partition (only when K = 1)')
parser.add_argument('--network', type=str, default='iresnet',
                    help='name of the neural network to train')
parser.add_argument('--kernel', type=int, default=7,#2,
                    help='kernel size')
parser.add_argument('--depth', type=int, default=5,#7,
                    help='network depth')
parser.add_argument('--n_layer', type=int, default=2,
                    help='(iresnet/resnet only) number of convolution per residual block')
parser.add_argument('--hidden', type=int, default=16,#128,
                    help='hidden dimension')
parser.add_argument('--hidden-classifier', type=int, default=32,
                    help='hidden dimension for final layer')
parser.add_argument('--max_hidden', type=int, default=32,#128,
                    help='(iresnet/resnet only) maximum hidden dimension')
parser.add_argument('--two_phase', action='store_true', default=False,
                    help='')
parser.add_argument('--print_every', type=int, default=-1,
                    help='')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for network seed and random partition')
parser.add_argument('--cudnn_deterministic', action='store_true', default=True,#False,
                    help='')
parser.add_argument('--min_sample', type=int, default=20,#0,
                    help='minimum number of pre-segmented light curve per class')
parser.add_argument('--max_sample', type=int, default=100000,
                    help='maximum number of pre-segmented light curve per class during testing')
parser.add_argument('--test', action='store_true', default=False,
                    help='test pre-trained model')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='continue training from checkpoint')
parser.add_argument('--no-log', action='store_true', default=False,
                    help='continue training from checkpoint')
parser.add_argument('--note', type=str, default='',
                    help='')
parser.add_argument('--project-name', type=str, default='',
                    help='for weights and biases tracking')
parser.add_argument('--decay-type', type=str, default='plateau',
                    help='')
parser.add_argument('--patience', type=int, default=5,
                    help='patience for learning decay')
parser.add_argument('--early_stopping', type=int, default=0,
                    help='terminate training if loss does not improve by 10% after waiting this number of epochs')
parser.add_argument('--embedding', type=str, default='data/ogle.unitsphere.pickle',
                    help='path to pickle file with embedding')
parser.add_argument('--classnames', type=str, default='data/ogle_class_names.txt',
                    help='path to text file with classnames')
parser.add_argument('--hierarchy', type=str, default='data/ogle.parent-child.txt',
                    help='path to text file with parent child relationships')
parser.add_argument('--classification', type=str, default='hidden',
                    help='how to compute classification from embedding (none, hidden or cosine)')
parser.add_argument('--loss', type=str, default='inv_corr',
                    help='how to compute loss (inv_corr, height)')
parser.add_argument('--all_labels', action='store_true', default=True,
                    help='whether or not to include parent labels')
parser.add_argument('--optimize', action='store_true', default=False,
                    help='whether to do hyperparemeter optimization')
args = parser.parse_args()


def get_device(path):
    device = os.listdir(path)[0]
    os.remove(path+'/'+device)
    return device
        
if args.classification=='none' and (args.network == 'resnet' or args.network == 'iresnet'):
    save_name = '{}-{}-K{}-D{}-NL{}-H{}-MH{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}'.format(args.filename[:-4],
                                                                              args.network,
                                                                              args.kernel,
                                                                              args.depth,
                                                                              args.n_layer,
                                                                              args.hidden,
                                                                              args.max_hidden,
                                                                              args.L,
                                                                              int(args.varlen_train),
                                                                              args.input,
                                                                              args.lr,
                                                                              args.hidden_classifier,
                                                                              max(args.dropout, args.dropout_classifier),
                                                                              int(args.two_phase))
elif args.network == 'resnet' or args.network == 'iresnet':
    save_name = '{}-{}-K{}-D{}-NL{}-H{}-MH{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}-CL{}'.format(args.filename[:-4],
                                                                              args.network,
                                                                              args.kernel,
                                                                              args.depth,
                                                                              args.n_layer,
                                                                              args.hidden,
                                                                              args.max_hidden,
                                                                              args.L,
                                                                              int(args.varlen_train),
                                                                              args.input,
                                                                              args.lr,
                                                                              args.hidden_classifier,
                                                                              max(args.dropout, args.dropout_classifier),
                                                                              int(args.two_phase),
                                                                              args.classification)
else:
    save_name = '{}-{}-K{}-D{}-H{}-L{}-V{}-{}-LR{}-CLIP{}-DROP{}-TP{}-CL{}'.format(args.filename[:-4],
                                                                              args.network,
                                                                              args.kernel,
                                                                              args.depth,
                                                                              args.hidden,
                                                                              args.L,
                                                                              int(args.varlen_train),
                                                                              args.input,
                                                                              args.lr,
                                                                              args.clip,
                                                                              args.dropout,
                                                                              int(args.two_phase),
                                                                              args.classification)
from torch.multiprocessing import current_process
if current_process().name != 'MainProcess':
    if args.njob > 1 or args.ngpu > 1:
        path = 'device'+save_name+args.note
        device = get_device(path)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device[0])
else:
    print('save filename:')
    print(save_name)
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from resnet import Classifier as resnet
from itcn import Classifier as itcn
from itin import Classifier as itin
from rnn import Classifier as rnn
from data import MyDataset as MyDataset

from light_curve import LightCurve
from util import *
from class_hierarchy import ClassHierarchy
from _train import train, cosdistance
from collections import OrderedDict

print("AVAILABLE?:",torch.cuda.is_available())
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dint = torch.cuda.LongTensor
    map_loc = 'cuda'
else:
    assert args.ngpu == 1
    dtype = torch.FloatTensor
    dint = torch.LongTensor
    map_loc = 'cpu'

if args.cudnn_deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if 'asassn' in args.filename:
    args.max_sample = 20000
if args.n_test == 1:
    lengths = [args.L]
else:
    lengths = np.linspace(16, args.L * 2, args.n_test).astype(np.int)
    if args.L not in lengths:
        lengths = np.sort(np.append(lengths, args.L))
data = joblib.load('data/{}'.format(args.filename))
with open(args.embedding, 'rb') as pf:
    embedding = pickle.load(pf)
    embed_labels = embedding['ind2label']
    linear_labels = embedding['label2ind'] 
    embedding = embedding['embedding']
    
# sanity check on dataset
for lc in data:
    positive = lc.errors > 0
    positive *= lc.errors < 99
    lc.times = lc.times[positive]
    lc.measurements = lc.measurements[positive]
    lc.errors = lc.errors[positive]
    lc.label = lc.label.replace(" ","")
    lc.p = abs(float(lc.p))

if 'macho' in args.filename:
    for lc in data:
        if 'LPV' in lc.label:
            lc.label = "LPV"

# Generate a list all labels for train/test split
unique_label, count = np.unique([lc.label for lc in data], return_counts=True)
use_label = unique_label[count >= args.min_sample]

all_label_string = [lc.label for lc in data]
unique_label, count = np.unique(all_label_string, return_counts=True)
print('------------before segmenting into L={}------------'.format(args.L))

label2int, int2label, unique_labels = {}, {}, OrderedDict()
with open(args.classnames) as class_list:
    for c in class_list:  
        num, classname = c.strip().split()
        label2int[classname] = int(num)
        int2label[int(num)] = classname
        unique_labels[classname] = None 
#unique_label, count = np.unique(list(unique_labels.keys()), return_counts=True) 
#convert_label = dict(zip(unique_label, np.arange(len(unique_label))))
#all_labels = np.array([convert_label[lc.label] for lc in data])
hierarchy = ClassHierarchy.from_file(args.hierarchy, id_type = int)

if args.all_labels and args.classification != 'none':
    unique_label, count = np.unique(list(unique_labels.keys()), return_counts=True) 
    use_label = unique_label
    convert_label = dict(zip(unique_label, np.arange(len(unique_label))))
    all_labels = np.array([convert_label[lc.label] for lc in data])
    convert_ind = dict(zip(np.arange(len(unique_label)), unique_label))
    convert_emb2ind = lambda i: convert_label[int2label[embed_labels[i]]] 
else:
    convert_label = dict(zip(use_label, np.arange(len(use_label))))
    all_labels = np.array([convert_label[lc.label] for lc in data])
    convert_ind = dict(zip(np.arange(len(use_label)), use_label))
    convert_emb2ind = lambda i: convert_label[int2label[embed_labels[i]]] if int2label[embed_labels[i]] in use_label else None

n_classes = len(unique_label)#len(use_label)
new_data = []
for cls in use_label:
    class_data = [lc for lc in data if lc.label == cls]
    new_data.extend(class_data)#[:min(len(class_data), args.max_sample)])
data = new_data

if args.input in ['dtdfg', 'dtfg', 'dtfe']:
    n_inputs = 3
elif args.input in ['df', 'f', 'g']:
    n_inputs = 1
else:
    n_inputs = 2


def get_network(n_classes, embedding, hidden=args.hidden, max_hidden=args.max_hidden, depth=args.depth, kernel=args.kernel, n_layer=args.n_layer, hidden_classifier=args.hidden_classifier):
    if args.network == 'itcn':
        clf = itcn(num_inputs=n_inputs, num_channels=[args.hidden] * args.depth, num_class=n_classes, 
                   num_outputs=embedding.shape[1], classification=args.classification, hidden=32,
                   dropout=args.dropout, kernel_size=args.kernel, dropout_classifier=0, aux=3,
                   padding='cyclic').type(dtype)
    elif args.network == 'tcn':
        clf = itcn(num_inputs=n_inputs, num_channels=[args.hidden] * args.depth, num_class=n_classes, 
                   num_outputs=embedding.shape[1], classification=args.classification, hidden=32,
                   dropout=args.dropout, kernel_size=args.kernel, dropout_classifier=0, aux=3,
                   padding='zero').type(dtype) 
    elif args.network == 'itin':
        clf = itin(num_inputs=n_inputs, kernel_sizes=[args.kernel,args.kernel+2,args.kernel+4],
                   num_channels=[args.hidden] * args.depth, num_class=n_classes, hidden=32, dropout=args.dropout,
                   dropout_classifier=0, aux=3, padding='cyclic', num_outputs=embedding.shape[1], 
                   classification=args.classification).type(dtype)
    elif args.network == 'tin':
        clf = itin(num_inputs=n_inputs, kernel_sizes=[args.kernel,args.kernel+2,args.kernel+4],
                   num_channels=[args.hidden] * args.depth, num_class=n_classes, hidden=32,
                   dropout=args.dropout, dropout_classifier=0, aux=3, padding='zero', num_outputs=embedding.shape[1],                               
                   classification=args.classification).type(dtype)
  #  elif args.classification == 'none' and args.network == 'iresnet':
  #      clf = resnet_original(n_inputs, n_classes, depth=args.depth, nlayer=args.n_layer, kernel_size=args.kernel,
  #                   hidden_conv=args.hidden, max_hidden=args.max_hidden, padding='cyclic',min_length=args.min_maxpool,
  #                   aux=3, dropout_classifier=args.dropout_classifier, hidden=hidden_classifier).type(dtype)
    elif args.network == 'iresnet':
        clf = resnet(n_inputs, n_classes, depth=depth, nlayer=n_layer, kernel_size=kernel,
                     hidden_conv=hidden, max_hidden=max_hidden, padding='cyclic',min_length=args.min_maxpool,
                     aux=3, dropout_classifier=args.dropout_classifier, hidden=hidden_classifier,
                     num_outputs=embedding.shape[1], classification=args.classification).type(dtype)
    elif args.network == 'resnet':
        clf = resnet(n_inputs, n_classes, depth=args.depth, nlayer=args.n_layer, kernel_size=args.kernel,
                     hidden_conv=args.hidden, max_hidden=args.max_hidden, padding='zero',min_length=args.min_maxpool,
                     aux=3, dropout_classifier=0, hidden=32, num_outputs=embedding.shape[1], 
                     classification=args.classification).type(dtype)
    elif args.network == 'gru':
        clf = rnn(num_inputs=n_inputs, hidden_rnn=args.hidden, num_layers=args.depth, num_class=n_classes, hidden=32,
                  rnn='GRU', dropout=args.dropout, aux=3, num_outputs=embedding.shape[1], 
                  classification=args.classification).type(dtype)
    elif args.network == 'lstm':
        clf = rnn(num_inputs=n_inputs, hidden_rnn=args.hidden, num_layers=args.depth, num_class=n_classes, hidden=32,
                  rnn='LSTM', dropout=args.dropout, aux=3, num_outputs=embedding.shape[1], 
                  classification=args.classification).type(dtype)
    return clf

def train_helper(param):
    global map_loc
    train_index, test_index, name = param
    split = [chunk for i in train_index for chunk in data[i].split(args.L, args.L) if data[i].label is not None]
    for lc in split:
        lc.period_fold()
    unique_label, count = np.unique([lc.label for lc in split], return_counts=True)
    print('------------after segmenting into L={}------------'.format(args.L))
    print(unique_label)
    print(count)

    # shape: (N, L, 3)
    X_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    label = np.array([convert_label[chunk.label] for chunk in split])

    x, means, scales = getattr(PreProcessor, args.input)(np.array(X_list), periods)
    print('shape of the training dataset array:', x.shape)
    mean_x = x.reshape(-1, n_inputs).mean(axis=0)
    std_x = x.reshape(-1, n_inputs).std(axis=0)
    x -= mean_x
    x /= std_x
    if args.varlen_train:
        x = np.array(X_list)
    if args.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)
    # shape: (N, 3, L-1)

    aux = np.c_[means, scales, np.log10(periods)]
    aux_mean = aux.mean(axis=0)
    aux_std = aux.std(axis=0)
    aux -= aux_mean
    aux /= aux_std
    scales_all = np.array([np.append(mean_x, 0), np.append(std_x, 0), aux_mean, aux_std])
    if not args.varlen_train:
        scales_all = None
    else:
        np.save(name + '_scales.npy', scales_all)

    train_idx, val_idx = train_test_split(label, 1 - args.frac_valid, -1)
    if args.ngpu < 0:
        torch.cuda.set_device(int(-1*args.ngpu))
        map_loc = 'cuda:{}'.format(int(-1*args.ngpu))

    #print('Using ', torch.cuda.current_device())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    sys.stdout = sys.__stdout__
    
    class_sizes = {l: len(np.where(label[train_idx]==l)[0]) for l in np.unique(label[train_idx])}
    #class_sizes = [len(np.where(label[train_idx]==l)[0]) for l in np.unique(label[train_idx])]
    class_weights = np.array([1. / class_sizes[l] if l in class_sizes.keys() else 0 for l in range(len(unique_labels))])
    #print("CLASS SIZES:", class_sizes)
    #print("CLASSES: ", convert_label)
    #print("CLASS WEIGHTS:", class_weights)
    data_weights = np.array([class_weights[l] for l in label[train_idx]])
    data_weights = torch.from_numpy(data_weights)
    sampler = WeightedRandomSampler(data_weights.type('torch.DoubleTensor'), len(label[train_idx]))

    train_dset = MyDataset(x[train_idx], aux[train_idx], label[train_idx])
    val_dset = MyDataset(x[val_idx], aux[val_idx], label[val_idx])
    train_loader = DataLoader(train_dset, batch_size=args.train_batch, drop_last=True, sampler = sampler)
    val_loader = DataLoader(val_dset, batch_size=128, shuffle=False, drop_last=False)

    split = [chunk for i in test_index for chunk in data[i].split(args.L, args.L)]
    for lc in split:
        lc.period_fold()

    # shape: (N, L, 3)
    x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
    periods = np.array([lc.p for lc in split])
    x, means, scales = getattr(PreProcessor, args.input)(np.array(x_list), periods)

    # whiten data
    x -= mean_x
    x /= std_x
    if args.varlen_train:
        x = np.array(X_list)
    if args.two_phase:
        x = np.concatenate([x, x], axis=1)
    x = np.swapaxes(x, 2, 1)
    # shape: (N, 3, L)

    label = np.array([convert_label[chunk.label] for chunk in split])
    aux = np.c_[means, scales, np.log10(periods)]
    aux -= aux_mean
    aux /= aux_std

    test_dset = MyDataset(x, aux, label)
    test_loader = DataLoader(test_dset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)

    if not args.no_log:
        import wandb
        hyperparameter_defaults = dict(
                patience = 5
                )
        wandb.init(project=args.project_name, config=hyperparameter_defaults, name=name)
        config = wandb.config
    if args.optimize: 
        hidden = config.hidden
        max_hidden = config.max_hidden
        depth = config.depth
        kernel = config.kernel
        #dropout_classifier = config.dropout_classifier
        n_layer = args.n_layer #config.n_layer
        hidden_classifier = config.hidden_classifier
        mdl = get_network(n_classes, embedding, hidden, max_hidden, depth, kernel, n_layer, hidden_classifier)
    else:
        hidden = args.hidden
        max_hidden = args.max_hidden
        depth = args.depth
        kernel = args.kernel
        #dropout_classifier = args.dropout_classifier
        n_layer = args.n_layer
        hidden_classifier = args.hidden_classifier
        mdl = get_network(n_classes, embedding)
    #print("MAPLOC:",map_loc)
    #raise Exception("stop")
    #mdl = mdl.to(map_loc)
    if not args.no_log:
        wandb.watch(mdl)
    if not args.test:
        if args.retrain:
            mdl.load_state_dict(torch.load(name + '.pth', map_location=map_loc))
            args.lr *= 0.01
        optimizer = optim.Adam(mdl.parameters(), lr=args.lr)
        torch.manual_seed(args.seed)
        train(mdl, optimizer, train_loader, val_loader, test_loader, args.max_epoch, embedding, embed_labels, 
              linear_labels, label2int, convert_ind, args.classification, convert_emb2ind, hierarchy,map_loc, print_every=args.print_every, 
              save=True, filename=name+args.classification+args.note, patience=args.patience,
              early_stopping_limit=args.early_stopping, use_tqdm=True, scales_all=scales_all, clip=args.clip,
              retrain=args.retrain, decay_type=args.decay_type, monitor='accuracy', log=not args.no_log,
              perm=args.permute, loss_type=args.loss)

    # load the model with the best validation accuracy for testing on the test set
    if args.classification != 'none':
        mdl.load_state_dict(torch.load(name + args.classification + args.note + '.pth', map_location=map_loc))
    else: 
        print("NAME:",name + args.note + '.pth')
        mdl.load_state_dict(torch.load(name + args.note + '.pth', map_location=map_loc),strict=False)
    # Evaluate model on sequences of different length
    accuracy_length = np.zeros(len(lengths))
    accuracy_class_length = np.zeros(len(lengths))
    tanimoto = np.zeros((len(lengths),8))
    heights = np.zeros((len(lengths),8))
    pathlengths = np.zeros((len(lengths),8))
    similarities = np.zeros((len(lengths),8))
    true_similarities = np.zeros((len(lengths),8))
    fdlist =  []
    preds, truth, probs = [], [], []
    mdl.eval()
    with torch.no_grad():
        for j, length in enumerate(lengths):
            split = [chunk for i in test_index for chunk in data[i].split(length, length)]
            # num_chunks = np.array([len(data[i].split(length, length)) for i in test_index])
            # num_chunks = num_chunks[num_chunks != 0]
            # assert np.sum(num_chunks) == len(split)
            for lc in split:
                lc.period_fold()

            # shape: (N, L, 3)
            x_list = [np.c_[chunk.times, chunk.measurements, chunk.errors] for chunk in split]
            periods = np.array([lc.p for lc in split])
            x, means, scales = getattr(PreProcessor, args.input)(np.array(x_list), periods)

            # whiten data
            x -= mean_x
            x /= std_x
            if args.two_phase:
                x = np.concatenate([x, x], axis=1)
            x = np.swapaxes(x, 2, 1)
            # shape: (N, 3, L)

            label = np.array([convert_label[chunk.label] for chunk in split])
            aux = np.c_[means, scales, np.log10(periods)]
            aux -= aux_mean
            aux /= aux_std

            test_dset = MyDataset(x, aux, label)
            test_loader = DataLoader(test_dset, batch_size=128, shuffle=False, drop_last=False, pin_memory=True)
            softmax = torch.nn.Softmax(dim=1)
            predictions = []
            ground_truths = []
            probabilities = []
            predicted_embed = []
            for i, d in enumerate(test_loader):
                x, aux_, y = d
                if args.classification=='none':
                    logprob = mdl(x.type(dtype), aux_.type(dtype))
                    predictions.extend(list(np.argmax(softmax(logprob).detach().cpu(), axis=1)))
                    probabilities.extend(list(softmax(logprob).detach().cpu().numpy()))
                else: 
                    logprob1, logprob2 = mdl(x.type(dtype), aux_.type(dtype))
                    if args.classification=='hidden':
                        predictions.extend(list(np.argmax(softmax(logprob2).detach().cpu(), axis=1)))
                        probabilities.extend(list(softmax(logprob2).detach().cpu().numpy()))
                    else:
                        predictions.extend(list(map(convert_emb2ind, np.argmax(softmax(cosdistance(embedding,logprob1)).detach().cpu(), axis=1))))
                        #predictions.extend(list(map(find_c_label, softmax(cosdistance(embeddings,logprob1)).detach().cpu())))
                        probabilities.extend(list(softmax(cosdistance(embedding,logprob1)).detach().cpu().numpy()))
                    predicted_embed.extend(logprob1)
                ground_truths.extend(list(y.numpy()))
                

            predictions = np.array(predictions)
            ground_truths = np.array(ground_truths)
            probabilities = np.array(probabilities)
            preds.append(predictions)
            truth.append(ground_truths)
            probs.append(probabilities)
            # pred_perobj = [np.argmax(np.log(probs[sum(num_chunks[:i]):sum(num_chunks[:i + 1])]).sum(axis=0))
            #                for i in range(len(num_chunks))]
            # gt_perobj = [ground_truths[sum(num_chunks[:i])] for i in range(len(num_chunks))]
            #

            # if len(lengths) == 1:
            #     np.save('{}_predictions.npy'.format(name), np.c_[predictions, ground_truths])
            #     np.save('{}_predictions_perobj.npy'.format(name), np.c_[pred_perobj, gt_perobj])
            #     np.save('{}_labels.npy'.format(name), use_label)

            accuracy_length[j] = (predictions == ground_truths).mean()
            accuracy_class_length[j] = np.array(
                [(predictions[ground_truths == l] == ground_truths[ground_truths == l]).mean()
                 for l in np.unique(ground_truths)]).mean()
            def find_distances(predictions, ground_truths): 
                #find_tanimoto = lambda i: hierarchy.metric2(int(label2int[convert_ind[predictions[i]]]), int(label2int[convert_ind[ground_truths[j]]]))
                def find_tanimoto(i):
                    #print(convert_ind[predictions[i]],convert_ind[ground_truths[j]])
                    return hierarchy.metric2(int(label2int[convert_ind[predictions[i]]]), int(label2int[convert_ind[ground_truths[i]]]))
                find_height = lambda i: hierarchy.lcs_height(label2int[convert_ind[predictions[i]]], label2int[convert_ind[ground_truths[i]]])
                find_pathlength = lambda i: hierarchy.pathlength(label2int[convert_ind[predictions[i]]], label2int[convert_ind[ground_truths[i]]])
                find_similarity = lambda i: np.matmul(embedding[linear_labels[label2int[convert_ind[predictions[i]]]]].T,embedding[linear_labels[label2int[convert_ind[ground_truths[i]]]]])
                find_true_similarity = lambda i: np.matmul(predicted_embed[i].T,embedding[linear_labels[label2int[convert_ind[ground_truths[i]]]]]) if args.classification !='none' else 0
                #s=[find_similarity(i) for i in range(len(predictions))]
                #print(s)
                #print(np.asarray(s))
                return np.asarray([find_tanimoto(i) for i in range(len(predictions))]), np.asarray([find_height(i) for i in range(len(predictions))]), np.asarray([find_pathlength(i) for i in range(len(predictions))]), np.asarray([find_similarity(i) for i in range(len(predictions))]), np.asarray([find_true_similarity(i) for i in range(len(predictions))])
            fd = find_distances(predictions, ground_truths)
            fdlist.append(fd)
            print("CONVERT IND:", convert_ind)
            tanimoto[j], heights[j], pathlengths[j], similarities[j], true_similarities[j] = np.concatenate(([[i] for i in np.mean(fd,axis=-1)],
                                                                                                            [[i] for i in np.median(fd,axis=-1)],
                                                                                                            [[i] for i in np.amin(fd,axis=-1)],
                                                                                                            [[i] for i in np.amax(fd,axis=-1)],
                                                                                                            [[i] for i in np.percentile(fd,1,axis=-1)],
                                                                                                            [[i] for i in np.percentile(fd,5,axis=-1)],
                                                                                                            [[i] for i in np.percentile(fd,95,axis=-1)],
                                                                                                            [[i] for i in np.percentile(fd,99,axis=-1)]),
                                                                                                            axis=-1)
    if args.ngpu > 1:
        return_device(path, device)
    return accuracy_length, accuracy_class_length, tanimoto[:,0], tanimoto[:,1],tanimoto[:,2],tanimoto[:,3],tanimoto[:,4],tanimoto[:,5],tanimoto[:,6],tanimoto[:,7], heights[:,0],heights[:,1],heights[:,2],heights[:,3],heights[:,4],heights[:,5],heights[:,6],heights[:,7], pathlengths[:,0],pathlengths[:,1],pathlengths[:,2],pathlengths[:,3],pathlengths[:,4],pathlengths[:,5],pathlengths[:,6],pathlengths[:,7], similarities[:,0],similarities[:,1],similarities[:,2],similarities[:,3],similarities[:,4],similarities[:,5],similarities[:,6],similarities[:,7], true_similarities[:,0],true_similarities[:,1],true_similarities[:,2],true_similarities[:,3],true_similarities[:,4],true_similarities[:,5], true_similarities[:,6],true_similarities[:,7],preds, truth, probs, fdlist


if __name__ == '__main__':

    jobs = []
    np.random.seed(args.seed)
    for i in range(args.K):
        if args.K == 1:
            i = args.pseed
        trains, tests = train_test_split(all_labels, train_size=args.frac_train, random_state=i)
        jobs.append((trains, tests, '{}/{}-{}'.format(args.path, save_name, i)))
    try:
        os.mkdir(args.path)
    except:
        pass
    if args.ngpu <= 1 and args.njob == 1:
        results = []
        for j in jobs:
            results.append(train_helper(j))
    else:
        create_device('device'+save_name+args.note, args.ngpu, args.njob)
        ctx = mp.get_context('spawn')
        with ctx.Pool(args.ngpu * args.njob) as p:
            results = p.map(train_helper, jobs)
        shutil.rmtree('device' + save_name+args.note)
    results = np.array(results)
    #results_all = np.c_[lengths, results[:, 0, :].T,results[:, 2, :].T,results[:, 3, :].T,results[:, 4, :].T, results[:, 5, :].T, results[:, 6, :].T]
    #results_class = np.c_[lengths, results[:, 1, :].T]
    #pred_truth = np.c_[lengths, results[:, 7, :].T, results[:, 8, :].T, results[:, 9, :].T]
    
    results_all = np.c_[lengths, results[:, 0, :].T]
    results_eval = results[:,2:42,:]
    results_class = np.c_[lengths, results[:, 1, :].T]
    pred_truth = results[:, 42:45, :].T#np.c_[lengths, results[:, 38:, :].T, results[:, 8, :].T]#, results[:, 9, :].T]
    results_raweval = results[:, 45, :] 
    print("INT2LABEL:",int2label)
    np.save('{}/{}{}-results.npy'.format(args.path, save_name, args.note), results_all)
    np.save('{}/{}{}-results-class.npy'.format(args.path, save_name, args.note), results_class)
    np.save('{}/{}{}-results-predtruth.npy'.format(args.path, save_name, args.note), pred_truth)
    np.save('{}/{}{}-results-eval.npy'.format(args.path, save_name, args.note), results_eval)
    np.save('{}/{}{}-results-raweval.npy'.format(args.path, save_name, args.note), results_raweval)
