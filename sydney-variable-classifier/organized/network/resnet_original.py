#Modified version of https://github.com/kmzzhang/periodicnetwork
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from padding import wrap


class MyMaxPool1d(nn.Module):
    def __init__(self, window, stride=1, min_length=-1, padding='zero'):
        """
        Custom MaxPool module to accomodate cyclic-permutation invariance and minimum seqeunce length requirements

        Parameters
        ----------
        window: int
            MaxPool window size
        stride: int
            MaxPool stride
        min_length: int
            Minimum sequence length for which MaxPool is applied
        padding: str
            "cyclic": symmetry padding for invariance
            "zero": zero padding for ordinary Cartesian network
        """
        super(type(self), self).__init__()
        self.window = window
        self.maxpool = nn.MaxPool1d(window, stride=stride)
        self.min_length = min_length
        self.padding = padding

    def forward(self, x):
        if x.shape[-1] >= self.min_length:
            mod = x.shape[2] % self.window
            if mod != 0 and self.padding != 'zero':
                x = torch.cat([x[:, :, -(self.window-mod):]] + [x], dim=2)
            return self.maxpool(x)
        else:
            return x

class Classifier(nn.Module):
    def __init__(self, num_inputs, num_class, depth=9, nlayer=2, kernel_size=3, hidden_conv=32, max_hidden=256,
                 aux=0, dropout=0, dropout_classifier=0, hidden=32, min_length=-1,padding='zero'):
        """
        Cyclic-permutation invariant 1D ResNet Classifier

        Parameters
        ----------
        num_inputs: int
            dimension of input seqeunce
        num_class: int
            number of classes
        depth: int
            number of residual modules used in the resnet
        nlayer: int
            number of convolutions in one residual module
        kernel_size: int
            kernel size
        hidden_conv: int
            hidden dimension for the first residual block, which doubles for each additional block
        max_hidden: int
            maximum hidden dimension for each residual block
        aux: int
            number of auxiliary inputs
        dropout: float
            dropout rate
        dropout_classifier: float
            dropout rate for the final MLP classifier
        hidden: int
            hidden dimension of the final two layer MLP classifier
        min_length: int
            minimum length for which maxpooling is applied (see docs for MyMaxPool1d)
        padding: str
            "cyclic": symmetry padding for invariance
            "zero": zero padding for ordinary Cartesian network
        """
        super(type(self), self).__init__()
        network, network2 = list(), list()
        self.a = ResBlock(num_inputs, hidden_conv, kernel_size, dropout, padding)
        for j in range(nlayer - 1):
            network.append(ResBlock(hidden_conv, hidden_conv, kernel_size, dropout, padding))
        for i in range(depth - 1):
            h0 = min(max_hidden, hidden_conv * 2 ** i)
            h = min(max_hidden, hidden_conv * 2 ** (i + 1))
            network2.append(MyMaxPool1d(2, stride=2, min_length=min_length, padding=padding))
            network2.append(ResBlock(h0, h, kernel_size, dropout, padding))
            for j in range(nlayer-1):
                network2.append(ResBlock(h, h, kernel_size, dropout, padding))
        self.conv = nn.Sequential(*network)
        self.conv2 = nn.Sequential(*network2)
        self.linear1 = nn.Conv1d(h + aux, hidden, 1)
        self.linear2 = nn.Conv1d(hidden, num_class, 1)
        self.dropout = nn.Dropout(dropout_classifier)
        self.aux = aux
        if aux > 0:
            self.linear = nn.Sequential(self.linear1, self.dropout, nn.ReLU(), self.linear2)
        else:
            self.linear = nn.Conv1d(h, num_class, 1)
        self.feature = padding

    def forward(self, x, aux=None):
        # N D L
        f = self.a(x)
        print("SHAPES:",f.size())
        f = self.conv(f)
        print(f.size())
        feature = self.conv2(f)
        self.cache = feature
        if self.aux > 0:
            feature = torch.cat((feature, aux[:,:,None].expand(-1,-1,feature.shape[2])), dim=1)
        logprob_ = self.linear(feature)
        if self.feature == 'zero':
            logprob = logprob_[:, :, -1]
        else:
            logprob = logprob_.mean(dim=2)
        raise Exception("Finished")
        return logprob


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dropout=0, padding='zero'):
        super(type(self), self).__init__()
        if padding == 'cyclic':
            net = [wrap(k-1, mode=padding),
                   weight_norm(nn.Conv1d(in_ch, out_ch, k)),
                   nn.ReLU(),
                   nn.Dropout(dropout),
                   wrap(k-1, mode=padding),
                   weight_norm(nn.Conv1d(out_ch, out_ch, k)),
                   nn.ReLU(),
                   nn.Dropout(dropout)]
        if padding == 'zero':
            net = [weight_norm(nn.Conv1d(in_ch, out_ch, k, padding=int(k/2))),
                   nn.ReLU(),
                   nn.Dropout(dropout),
                   weight_norm(nn.Conv1d(out_ch, out_ch, k, padding=int(k/2))),
                   nn.ReLU(),
                   nn.Dropout(dropout)]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, dropout=0, padding='zero'):
        super(type(self), self).__init__()
        self.conv = ConvBlock(in_ch, out_ch, k, dropout, padding)
        if in_ch != out_ch:
            self.conv0 = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.conv0 = None
        if padding == 'zero' and k % 2 == 0:
            self.keepdims = True
        else:
            self.keepdims = False
            
        self.neql = in_ch != out_ch
        self.k = k
        self.pad = padding


    def forward(self, x):
        if self.neql:
            print("NOT equal")
        else: 
            print("Equal")
            
        y = self.conv(x)
        print("K:",self.k)
        print("Padding:",self.pad)
        if self.conv0 is not None:
            print("SkipShape:",self.conv0(x.float()).size())
        else: 
            print("NoSkipShape:",y.size())
        print("ConvShape:", y.size())
        
        if self.keepdims:
            y = y[:, :, 1:-1]
        if self.conv0 is not None:
            return y + self.conv0(x.float())
        return y + x