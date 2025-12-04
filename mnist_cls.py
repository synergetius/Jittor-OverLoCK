# classification mnist example 
import jittor as jt  # 将 jittor 引入
from jittor import nn, Module  # 引入相关的模块
import numpy as np
import sys, os
import random
import math 
from jittor import init
if jt.has_cuda:
    jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。
# 如果 jt.flags.use_cuda=1，表示使用GPU训练 如果 jt.flags.use_cuda = 0 表示使用 CPU
from jittor.dataset.mnist import MNIST 
#由于 MNIST 是一个常见的数据集，其数据载入已经被封装进 jittor 所以可以直接调用。
import matplotlib.pyplot as plt
import pylab as pl # 用于绘制 Loss 曲线 和 MNIST 数据

import gzip
from PIL import Image
from jittor.dataset import Dataset
from jittor_utils.misc import download_url_to_local

class MNIST(Dataset):
    def __init__(self, data_root="./mnist_data/", train=True ,download=True, batch_size=1, shuffle=False):
        # if you want to test resnet etc you should set input_channel = 3, because the net set 3 as the input dimensions
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = train
        if download == True:
            self.download_url()

        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        # this function must be called
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)
    def __getitem__(self, index):
        img = Image.fromarray (self.mnist['images'][index]) 
        img = np.array (img)
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32), self.mnist['labels'][index]

    def download_url(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]

        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)
            
class Model (Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.conv1 = nn.Conv (1, 32, 3, 1) # no padding
        
        self.conv2 = nn.Conv (32, 64, 3, 1)
        self.bn = nn.BatchNorm(64)

        self.max_pool = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (64 * 12 * 12, 256)
        self.fc2 = nn.Linear (256, 10)
    def execute (self, x) : 
        # it's simliar to forward function in Pytorch 
        x = self.conv1 (x)
        x = self.relu (x)
        
        x = self.conv2 (x)
        x = self.bn (x)
        x = self.relu (x)
        
        x = self.max_pool (x)
        x = jt.reshape (x, [x.shape[0], -1])
        x = self.fc1 (x)
        x = self.relu(x)
        x = self.fc2 (x)
        return x
def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        losses.append(loss.numpy()[0])
        losses_idx.append(epoch * lens + batch_idx)
        
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader) ,
                100. * batch_idx / len(train_loader), loss.numpy()[0]))

def val(model, val_loader, epoch):
    model.eval()
    
    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print(f'Test Epoch: {epoch} [{batch_idx}/{len(val_loader)}]\tAcc: {acc:.6f}')    	
        print('Test Acc =', total_acc / total_num)
    
batch_size = 64
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
epochs = 1
losses = []
losses_idx = []
train_loader = MNIST(train=True, batch_size=batch_size, shuffle=True)

val_loader = MNIST(train=False, batch_size=1, shuffle=False)

model = Model ()
optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
for epoch in range(epochs):
    train(model, train_loader, optimizer, epoch, losses, losses_idx)
    val(model, val_loader, epoch)
    
pl.plot(losses_idx, losses)
pl.xlabel('Iterations')
pl.ylabel('Train_loss')