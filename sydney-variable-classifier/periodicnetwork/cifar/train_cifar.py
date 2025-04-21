#Modified version of https://github.com/kmzzhang/periodicnetwork
import os
import argparse

parser = argparse.ArgumentParser(description='Periodic Permuted Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',default=True,
                    help='use CUDA (default: True)')
parser.add_argument('--periodic', action='store_true',default=False,
                    help='PP-MNIST if true; P-MNIST if false.')
parser.add_argument('--augment', action='store_true',default=False,
                    help='use data augmentation')
parser.add_argument('--dropout', type=float, default=0,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='depth of network')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--nhid_max', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--network', type=str, default='tcn',
                    help='network name itcn/tcn/itin/tin/iresnet/resnet/gru/lstm')
parser.add_argument('--path', type=str, default='results',
                    help='network')
parser.add_argument('--project', type=str, default='none',
                    help='for weights and biases tracking')
parser.add_argument('--ngpu', type=str, default='0',
                    help='gpu device number to use when gpu_count > 1')
parser.add_argument('--name', type=str, default='none',
                    help='save filename')
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

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"]=args.ngpu
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from data_cifar import data_generator
from resnet_3D import Classifier as resnet
from itcn import Classifier as itcn
from itin import Classifier as itin
from rnn import Classifier as rnn
from collections import OrderedDict
from class_hierarchy import ClassHierarchy
from torch import from_numpy
import numpy as np
import pickle
from fuzzywuzzy import fuzz, process
import scipy.spatial.distance as distance
import wandb

if 'tcn' not in args.network:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

root = './data/mnist'
name = args.path + '/{}_{}_{}_{}_{}'.format(args.network, args.nhid, args.nhid_max, args.dropout, args.levels)
print(name)
wandb.init(project=args.project, config=args, name=name)

batch_size = args.batch_size
#n_classes = 10
input_channels = 3
seq_length = int(3072 / input_channels)
#input_channels = 1
#seq_length = int(784 / input_channels)
epochs = 18 if args.network in ['iresnet','resnet'] else 40
steps = 0
test_accuracy = []
train_accuracy = []
final_test_accuracy = []

print(args)
train_loader, valid_loader, test_loader = data_generator(root, batch_size)

permute = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()
channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize

with open(args.embedding, 'rb') as pf:
    embedding = pickle.load(pf)
    embed_labels = embedding['ind2label']
    linear_labels = embedding['label2ind'] 
    embedding = embedding['embedding']
        
label2int, int2label, unique_labels = {}, {}, OrderedDict()
with open(args.classnames) as class_list:
    for c in class_list:  
        num, classname = c.strip().split()
        label2int[classname] = int(num)
        int2label[int(num)] = classname
        unique_labels[classname] = None 
classes = label2int.keys()
n_classes = len(classes)
            
if True:#args.all_labels and args.classification != 'none':
    unique_label, count = np.unique(list(unique_labels.keys()), return_counts=True) 
    use_label = unique_label
    convert_label = dict(zip(unique_label, np.arange(len(unique_label))))
    convert_ind = dict(zip(np.arange(len(unique_label)), unique_label))
    convert_emb2ind = lambda i: convert_label[int2label[embed_labels[i]]]
    
if args.network == 'itcn':
    model = itcn(input_channels, channel_sizes, n_classes, hidden=1, kernel_size=kernel_size, dropout=args.dropout,
                 padding='cyclic')
elif args.network == 'tcn':
    model = itcn(input_channels, channel_sizes, n_classes, hidden=1, kernel_size=kernel_size, dropout=args.dropout,
                 padding='zero')
elif args.network == 'iresnet':
    model = resnet(input_channels, n_classes, depth=args.levels, nlayer=9, kernel_size=kernel_size,
                   hidden_conv=args.nhid, max_hidden=args.nhid_max, padding='zero', aux=0,
                   dropout_classifier=args.dropout, hidden=32, num_outputs=embedding.shape[1], classification=args.classification)
elif args.network == 'resnet':
    model = resnet(input_channels, n_classes, depth=args.levels, nlayer=2, kernel_size=kernel_size,
                   hidden_conv=args.nhid, max_hidden=args.nhid_max, padding='zero', aux=0,
                   dropout_classifier=args.dropout, hidden=32)

if args.cuda:
    model.cuda()
    permute = permute.cuda()
wandb.watch(model)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dint = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dint = torch.LongTensor

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def find_embedding(embedding, linear_labels, classes, convert, convert_ind):
    def helper(class1): 
        new_class = process.extractOne(convert_ind[class1], classes, scorer=fuzz.token_set_ratio)[0]
        return embedding[linear_labels[convert[new_class]]]
    return helper
    
def inv_correlation(y_true, y_pred):
    """ Computes 1 minus the dot product between corresponding pairs of samples in two tensors. """
    return 1. - torch.sum(y_true * y_pred, axis = -1)

def cosdistance(embeddings,preds):
    preds=preds.detach()
    cos = torch.nn.CosineSimilarity(dim=0)
    def onepred(pred):
    #    #pred=pred.detach()
        oneembed = lambda embedding: 1-distance.cosine(embedding, pred)
        res = list(map(oneembed, embeddings))
        return res
    return  torch.Tensor(list(map(onepred, preds)))
    
find_embeds = find_embedding(embedding, linear_labels, classes, label2int, convert_ind)

def train(ep):
    softmax = torch.nn.Softmax(dim=1)
    hierarchy = ClassHierarchy.from_file(args.hierarchy, id_type = int)
    
    loss_fn1 = inv_correlation

    global steps
    train_accuracies = []
    train_loss = 0
    weight = 1
    model.train()
    predictions = []
    for batch_idx, (data, target) in enumerate(train_loader):
        print("DATA:",data.size())
    #    print("TARGET:",target)
        if args.cuda: data, target = data.cuda(), target.cuda()
        data = data.view(-1, input_channels, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
        #data = data[:, :, permute]
    #    print("DATA2:",data)
        if args.periodic:
            for i in range(data.shape[0]):
                start = np.random.randint(0, seq_length - 1)
                data[i] = torch.cat((data[i, :, start:], data[i, :, :start]), dim=1)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
       # print("INPT:",model(data),model(data)[0].shape,model(data)[1].shape)
        logprob1, logprob2 = model(data) #model(model(data))
            
        embeds = from_numpy(np.asarray([find_embeds(i.item()) for i in target.type(dint)]))
        print("Embeds:",embeds, embeds.size())
        print("logprob:",logprob1, logprob1.size())
        loss1 = torch.mean(loss_fn1(embeds, logprob1))
        if args.classification == 'hidden':
            loss2 = weight*torch.nn.CrossEntropyLoss()(logprob2, target.type(dint))
            #weight*torch.nn.CrossEntropyLoss(torch.argmax(softmax(logprob2),dim=-1), target)
            loss = loss1 + loss2
        else:
            loss = loss1
        
        print("LOSS:",loss)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        
        if args.classification=='hidden':
            pred=np.argmax(softmax(logprob2).detach().cpu(), axis=1)
        else: 
            pred=list(map(convert_emb2ind,np.argmax(softmax(cosdistance(embedding,logprob1)).detach().cpu(), axis=1)))
        print("pred:",pred)
        print("target:",target)
        predictions.extend(pred)
                
        train_loss += loss.cpu().detach().numpy()
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            train_accuracy.append(train_loss.item()/args.log_interval)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss/args.log_interval, steps))
            train_accuracies.append(train_loss/args.log_interval)
            train_loss = 0

    return np.array(train_accuracies).mean()


def test(save=False):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)    
    hierarchy = ClassHierarchy.from_file(args.hierarchy, id_type = int)
    test_loss = 0
    correct = 0
    loss_fn1 = inv_correlation
    weight = 1
    global test_accuracy
    predictions = []
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
            #data = data[:, :, permute]
            if args.periodic:
                for i in range(data.shape[0]):
                    start = np.random.randint(0, seq_length - 1)
                    data[i] = torch.cat((data[i, :, start:], data[i, :, :start]), dim=1)
            data, target = Variable(data, volatile=True), Variable(target)
            
            logprob1, logprob2 = model(data) #model(model(data), dim=1)
            
            embeds = from_numpy(np.asarray([find_embeds(i.item()) for i in target.type(dint)]))
            loss1 = torch.mean(loss_fn1(logprob1, embeds))
                    
            if args.classification == 'hidden':
                loss2 = weight*torch.nn.CrossEntropyLoss()(logprob2, target.type(dint))
                test_loss = loss1 + loss2
            else:
                test_loss = loss1
            test_loss += test_loss.item()
            #output = F.log_softmax(model(data))
            
            if args.classification=='hidden':
                pred=np.argmax(softmax(logprob2).detach().cpu(), axis=1)
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            else: 
                pred=np.array(list(map(convert_emb2ind,np.argmax(softmax(cosdistance(embedding,logprob1)).detach().cpu(), axis=1))))
                print("Test pred:",pred)
                print("Test target:",np.array(target))
                print(pred == np.array(target))
                print((pred == np.array(target)).sum())
                correct += (pred == np.array(target)).sum()
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            predictions.extend(pred)

        test_loss /= len(test_loader.dataset)
        if args.classification=='hidden':
            test_accuracy.append(100. * correct.numpy() / len(test_loader.dataset))
        else: 
            test_accuracy.append(100. * correct / len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * float(correct) / len(test_loader.dataset)))
        return test_loss
    
def final_test(save=False):
    model.eval()
    softmax = torch.nn.Softmax(dim=1)    
    hierarchy = ClassHierarchy.from_file(args.hierarchy, id_type = int)
    test_loss = 0
    correct = 0
    loss_fn1 = inv_correlation
    weight = 1
    predictions = []
    global final_test_accuracy
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data = data.view(-1, input_channels, int(np.sqrt(seq_length)), int(np.sqrt(seq_length)))
            #data = data[:, :, permute]
            if args.periodic:
                for i in range(data.shape[0]):
                    start = np.random.randint(0, seq_length - 1)
                    data[i] = torch.cat((data[i, :, start:], data[i, :, :start]), dim=1)
            data, target = Variable(data, volatile=True), Variable(target)
            
            logprob1, logprob2 = model(data) #model(model(data), dim=1)
            
            embeds = from_numpy(np.asarray([find_embeds(i.item()) for i in target.type(dint)]))
            loss1 = torch.mean(loss_fn1(logprob1, embeds))
                    
            if args.classification == 'hidden':
                loss2 = weight*torch.nn.CrossEntropyLoss()(logprob2, target.type(dint))
                test_loss = loss1 + loss2
            else:
                test_loss = loss1
            test_loss += test_loss.item()
            #output = F.log_softmax(model(data))
            
            if args.classification=='hidden':
                pred=np.argmax(softmax(logprob2).detach().cpu(), axis=1)
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            else: 
                pred=np.array(list(map(convert_emb2ind,np.argmax(softmax(cosdistance(embedding,logprob1)).detach().cpu(), axis=1))))
                print("Test pred:",pred)
                print("Test target:",np.array(target))
                print(pred == np.array(target))
                print((pred == np.array(target)).sum())
                correct += (pred == np.array(target)).sum()
            predictions.extend(pred)
            #pred = output.data.max(1, keepdim=True)[1]
            #correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        if args.classification=='hidden':
            final_test_accuracy.append(100. * correct.numpy() / len(test_loader.dataset))
        else: 
            final_test_accuracy.append(100. * correct / len(test_loader.dataset))
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * float(correct) / len(test_loader.dataset)))
        np.save('results_pred.npy', predictions)
        np.save('results_truth.npy', target.cpu())
        return test_loss


if __name__ == "__main__":

    if args.network not in ['itcn', 'tcn']:
        every = 10
    else:
        every = 15
    epochs = int(epochs)

    for epoch in range(1, epochs+1):
        if not args.augment:
            np.random.seed(args.seed)
        train_loss = train(epoch)
        test_loss = test()
        wandb.log({"Train Loss": train_loss,
                   "Test Loss": test_loss,
                   "Test Acc": test_accuracy[-1]})
        if epoch % every == 0:
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    final_test_loss = final_test()
    np.save(name + '_test.npy', np.array(final_test_accuracy))
    np.save(name + '_valid.npy', np.array(test_accuracy))
    np.save(name + '_train.npy', np.array(train_accuracy))
