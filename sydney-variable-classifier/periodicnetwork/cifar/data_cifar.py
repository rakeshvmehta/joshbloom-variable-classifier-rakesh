#Modified vrsion of https://github.com/kmzzhang/periodicnetwork
import torch
from torchvision import datasets, transforms
import pickle

def data_generator(root, batch_size):
    cifar = pickle.load( open( "data/cifar_data.pkl", "rb" ) )
    dsize = len(cifar['data'])
    #cifar['data'] = cifar['data'][:, :len(cifar['data'][0])//3]
    train_set = [[cifar['data'][i], cifar['labels'][i]] for i in range(int(dsize//1.4))]# if cifar['labels'][i]<2]
    #[cifar['data'][:(len(cifar['data'])//3)], cifar['labels'][:(len(cifar['data'])//3)]]
    valid_set = [[cifar['data'][i], cifar['labels'][i]] for i in range(int(dsize//1.4), int(dsize//1.2))]#,len(cifar['data'])) if cifar['labels'][i]<2]
    #[cifar['data'][(len(cifar['data'])//3):], cifar['labels'][(len(cifar['data'])//3):]]
    test_set = [[cifar['data'][i], cifar['labels'][i]] for i in range(int(dsize//1.2),int(dsize))]

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(dsize)-int(dsize//1.2))
    return train_loader, valid_loader, test_loader
