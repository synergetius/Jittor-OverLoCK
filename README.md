# Jittor-OverLoCK

本项目基于Jittor框架，对CVPR 2025的论文[OverLoCK](https://github.com/LMMMEng/OverLoCK)进行预训练环节的实验复现，并编写PyTorch版本的程序进行对齐实验。

## 环境配置

在Windows系统中，由于兼容性问题安装Jittor失败，所以安装了WSL Ubuntu 22.04.5用于实验。

安装了anaconda 3用于管理python环境。

Jittor的训练环境使用python 3.7，使用官网提供的安装命令：

```bash
python -m pip install jittor
python -m jittor.test.test_example
python -m jittor.test.test_cudnn_op
```

PyTorch的训练环境根据OverLoCK官方版本的实现配置：

```bash
# Environments:
cuda==12.1
python==3.10
# Dependencies:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
pip install timm==0.6.12
pip install mmengine==0.2.0
```

由于`https://shi-labs.com/natten/wheels/`网络连接不佳，建议手动下载`natten`的`.whl`文件。



## 模型

由于计算资源有限，使用了一个比OverLoCK-XT更小规模的模型OverLoCK-XXT进行训练，参数配置如下：

```python
model = OverLoCK(
    depth = [1, 1, 2, 1],         
    sub_depth = [3, 1],              
    embed_dim = [24, 48, 96, 128],   
    kernel_size = [13, 11, 9, 7],      
    mlp_ratio = [2, 2, 2, 2],        
    sub_num_heads = [1, 2],          
    sub_mlp_ratio = [2, 2],
    projection = 256,               
    **kwargs
)
```



## 数据集

采用TinyImagenet数据集，它包含200个类别，训练集每个类别有500张图像，验证集每个类别有50张图像，尺寸均为64x64。由于计算资源和时间的限制，将类别序号按字典序排序后选出前20个类别进行训练和测试。



## 训练和测试

`train.py`是Jittor的训练和测试脚本，包含数据集加载、训练过程和验证集上的评估。

```python
import argparse
import datetime
import json
import csv
import pandas as pd
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
os.environ['JT_SYNC'] = '1'
os.environ['trace_py_var'] = '3'
from jittor.dataset import DataLoader, Dataset
from jittor.transform import Compose, Resize, RandomCrop, CenterCrop, ToTensor, RandomHorizontalFlip
from jittor import nn
import jittor
import models
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import numpy as np
from PIL import Image  
def get_args_parser():
    parser = argparse.ArgumentParser(description='Jittor Training', add_help=False)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)') 
    parser.add_argument('--aux-loss-ratio', type=float, default=0.4,
                        help='Aux loss weight')
    return parser
class TinyImageNet(Dataset):
    N_CLASS = 20
    N_TRAIN = 500 
    N_VAL = 50 
    def __init__(self, root, train=True, transform=None, debug=False):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.debug = debug

        if self.train:
            self.data_dir = os.path.join(self.root, 'train') if 'train' not in self.root else self.root
        else:
            self.data_dir = os.path.join(self.root, 'val', 'images')
            self.annotation_file = os.path.join(self.root, 'val', 'val_annotations.txt')

        self.samples = []
        if self.train:
            print(f"[DEBUG] 开始扫描训练集类别...")  
            self.classes = sorted(os.listdir(self.data_dir))[:self.N_CLASS]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"[DEBUG] 找到 {len(self.classes)} 个类别")  
            sample_ind = os.path.join(self.root, 'train_samples.csv')
            print('classes number:', len(self.classes))
            
            sample_count = np.zeros(len(self.classes))

            for i, cls in enumerate(self.classes):
                cls_dir = os.path.join(self.data_dir, cls, 'images')
                cls_idx = self.class_to_idx[cls]
                ls = sorted(os.listdir(cls_dir)) 
                ls = ls[:self.N_TRAIN]
                sample_count[i] += len(ls)
                for img_name in ls: 
                    self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
            with open(sample_ind, mode = "w") as f:
                csv.writer(f).writerows(self.samples)
            plot = False
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('train_samples.png', dpi=300, bbox_inches='tight')
                
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")  
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            self.class_to_img = {}
            for line in lines:
                img, cls = line.split('\t')[:2]
                self.class_to_img.setdefault(cls, []).append(img)
            
            self.classes = sorted(list(self.class_to_img.keys()))[:self.N_CLASS] 
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print('validation classes number:', len(self.classes))
            sample_count = np.zeros(len(self.classes))
            
            
            self.samples = []
            for cls in self.classes:
                imgs = self.class_to_img[cls]
                for img in imgs[:self.N_VAL]:
                    sample_count[self.class_to_idx[cls]] += 1
                    self.samples.append((os.path.join(self.data_dir, img), self.class_to_idx[cls]))
            plot = False
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('val_samples.png', dpi=300, bbox_inches='tight')
        self.set_attrs(total_len=len(self.samples))


    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, jittor.int32(label)
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, logfile):
    model.train()
    total_batches = len(loader)
    time0 = time.time()
    losses_m = 0
    losses_aux_m = 0
    total = 0 
    for i, (input, target) in enumerate(loader):
        try:
            output = model(input)
            output_main = output['main']
            output_aux = output['aux']
            loss_main = loss_fn(output_main, target)
            loss_aux = loss_fn(output_aux, target)
            loss = loss_main + args.aux_loss_ratio * loss_aux
            losses_m += loss * input.size(0) 
            losses_aux_m += loss_aux * input.size(0)
            optimizer.step(loss)
            print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")
            print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}", file = logfile)
            total += input.size(0)
            jittor.sync_all()
            jittor.gc()
        except Exception as e:
            print(f"训练第{i+1}个 batch 时出错: {e}")
            print(f"训练第{i+1}个 batch 时出错: {e}", file = logfile)
            raise
    if total:
        losses_m /= total
        losses_aux_m /= total
    epoch_time = time.time() - time0
    train_info_dict = {
        'epoch_time': round(epoch_time / 60, 1),
        'loss': float(losses_m),
        'loss_aux': float(losses_aux_m)
    }
    
    return OrderedDict(train_info_dict)
def correct_topk(output, target, k = 1):
    _, predicted = jittor.topk(output, k, dim = 1)
    correct = (predicted == target).sum().float().item()
    return correct
def validate(epoch, model, loader, loss_fn, args, logfile):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_batches = len(loader)
    losses_m = 0
    total = 0
    with jittor.no_grad():
        for i, (input, target) in enumerate(loader):
            output = model(input)
            loss = loss_fn(output, target)
            losses_m += loss * input.size(0)
            total += input.size(0)
            correct_top1 += correct_topk(output, target, 1)
            correct_top5 += correct_topk(output, target, 5)
            print(f"[Epoch {epoch+1} | validated Batch {i+1}/{total_batches}]")
            print(f"[Epoch {epoch+1} | validated Batch {i+1}/{total_batches}]", file = logfile)
    losses_m /= total
    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total

    metrics = OrderedDict([('loss', float(losses_m)),
                           ('acc_top1', float(acc_top1)),
                           ('acc_top5', float(acc_top5))])

    return metrics
def main(args):
    jittor.flags.use_cuda = 1
    jittor.flags.log_silent = 0
    jittor.flags.log_v = 1
    logfile = open("log.txt", "w")
    def custom_normalize(tensor):
        mean = jittor.array([0.485, 0.456, 0.406]).view(3,1,1)
        std = jittor.array([0.229, 0.224, 0.225]).view(3,1,1)
        return (tensor - mean) / std
    
    train_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = True,
        transform = Compose([
            Resize(64),
            RandomHorizontalFlip(),  
            ToTensor(),
            custom_normalize
        ]),
        debug = True
    )
    val_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = False,
        transform = Compose([
            Resize(64),
            ToTensor(),
            custom_normalize
        ]) 
    )
    train_loader = train_dataset.set_attrs(batch_size = args.batch_size, shuffle = True) 
    val_loader = val_dataset.set_attrs(batch_size = args.batch_size, shuffle=False) 
    load_model = False
    MODEL_PATH = "overlock.pkl"
    model = models.overlock_xxt(num_classes = TinyImageNet.N_CLASS)
    if load_model:
        model.load_state_dict(jittor.load(MODEL_PATH))
    
    optimizer = jittor.optim.AdamW(model.parameters(), lr = 1e-3) 
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0 
    num_epochs = 20 
    
    train_loss = []
    train_loss_aux = []
    epoch_time = []
    val_loss = []
    acc_top1 = []
    acc_top5 = []
    for epoch in range(start_epoch, num_epochs):

        train_metrics = train_one_epoch(
            epoch, model, train_loader, 
            optimizer, train_loss_fn, args, logfile
        )
        train_loss.append(train_metrics['loss'])
        train_loss_aux.append(train_metrics['loss_aux'])
        epoch_time.append(train_metrics['epoch_time'])
        
        
        val_metrics = validate(epoch, model, val_loader, validate_loss_fn, args, logfile)
        val_loss.append(val_metrics["loss"])
        acc_top1.append(val_metrics["acc_top1"])
        acc_top5.append(val_metrics["acc_top5"])
        print(f"[Epoch {epoch+1}")
        print(f"[Epoch {epoch+1}", file = logfile)
        for key, value in train_metrics.items():
            print(f"{key}: {value}")
            print(f"{key}: {value}", file = logfile)
        for key, value in val_metrics.items():
            print(f"{key}: {value}")
            print(f"{key}: {value}", file = logfile)
        print("]")
        print("]", file = logfile)
    log = {
        "train_loss":train_loss, 
        "train_loss_aux":train_loss_aux,
        "epoch_time":epoch_time,
        "val_loss":val_loss,
        "acc_top1":acc_top1,
        "acc_top5":acc_top5
    }
    pd.DataFrame(log).to_csv("log.csv", index = False, header = True)
    jittor.save(model.state_dict(), MODEL_PATH)
    logfile.close()
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
    
    
```



`train_torch.py`是PyTorch的训练和测试脚本，实现基本相同的训练和测试流程。

```python
import argparse
import datetime
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, ToTensor, RandomHorizontalFlip
import models_torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg') 

from torch.utils.data import Dataset, DataLoader
from PIL import Image 

device = torch.device("cuda")

def get_args_parser():
    parser = argparse.ArgumentParser(description='Torch Training', add_help=False)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--aux-loss-ratio', type=float, default=0.4,
                        help='Aux loss weight')
    return parser
class TinyImageNet(Dataset):
    N_CLASS = 20 
    N_TRAIN = 500 
    N_VAL = 50 
    def __init__(self, root, train=True, transform=None, debug=False):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.debug = debug

        if self.train:
            self.data_dir = os.path.join(self.root, 'train') if 'train' not in self.root else self.root
        else:
            self.data_dir = os.path.join(self.root, 'val', 'images')
            self.annotation_file = os.path.join(self.root, 'val', 'val_annotations.txt')

        self.samples = []
        if self.train:
            print(f"[DEBUG] 开始扫描训练集类别...")  
            self.classes = sorted(os.listdir(self.data_dir))[:self.N_CLASS]
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"[DEBUG] 找到 {len(self.classes)} 个类别")  
            sample_ind = os.path.join(self.root, 'train_samples.csv')

            print('classes number:', len(self.classes))
            
            sample_count = np.zeros(len(self.classes))

            for i, cls in enumerate(self.classes):
                cls_dir = os.path.join(self.data_dir, cls, 'images')
                cls_idx = self.class_to_idx[cls]
                ls = sorted(os.listdir(cls_dir)) 
                ls = ls[:self.N_TRAIN]
                sample_count[i] += len(ls)
                for img_name in ls: 
                    self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
            plot = False
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('train_samples.png', dpi=300, bbox_inches='tight')
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")  
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()

            self.class_to_img = {}
            for line in lines:
                img, cls = line.split('\t')[:2]
                self.class_to_img.setdefault(cls, []).append(img)
            
            self.classes = sorted(list(self.class_to_img.keys()))[:self.N_CLASS] ##########
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print('validation classes number:', len(self.classes))
            sample_count = np.zeros(len(self.classes))
            
            
            self.samples = []
            for cls in self.classes:
                imgs = self.class_to_img[cls]
                for img in imgs[:self.N_VAL]:
                    sample_count[self.class_to_idx[cls]] += 1
                    self.samples.append((os.path.join(self.data_dir, img), self.class_to_idx[cls]))
            plot = False
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('val_samples.png', dpi=300, bbox_inches='tight')

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img.to(device), torch.tensor(label, dtype = torch.int64).to(device)
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, logfile):
    model.train()
    total_batches = len(loader)
    time0 = time.time()
    losses_m = 0
    losses_aux_m = 0
    total = 0 
    for i, (input, target) in enumerate(loader):
        try:
            optimizer.zero_grad()
            
            output = model(input)
            output_main = output['main']
            output_aux = output['aux']
            loss_main = loss_fn(output_main, target)
            loss_aux = loss_fn(output_aux, target)
            loss = loss_main + args.aux_loss_ratio * loss_aux
            loss.backward()
            
            losses_m += loss * input.size(0) 
            losses_aux_m += loss_aux * input.size(0)
            
            optimizer.step()
       
            print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")
            print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}", file = logfile)
            total += input.size(0)
        except Exception as e:
            print(f"训练第{i+1}个 batch 时出错: {e}")
            print(f"训练第{i+1}个 batch 时出错: {e}", file = logfile)
            raise
    if total:
        losses_m /= total
        losses_aux_m /= total
    epoch_time = time.time() - time0
    train_info_dict = {
        'epoch_time': round(epoch_time / 60, 1),
        'loss': float(losses_m),
        'loss_aux': float(losses_aux_m)
    }
    return OrderedDict(train_info_dict)
def correct_topk(output, target, k = 1):
    _, predicted = torch.topk(output, k, dim = 1)
    correct = (predicted == target.unsqueeze(-1)).sum().float().item()
    return correct
def validate(epoch, model, loader, loss_fn, args, logfile):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_batches = len(loader)
    losses_m = 0
    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            output = model(input)
            tgt = F.one_hot(target, num_classes = TinyImageNet.N_CLASS).to(torch.float)
            loss = loss_fn(output, tgt)
            losses_m += loss * input.size(0)
            total += input.size(0)
            correct_top1 += correct_topk(output, target, 1)
            correct_top5 += correct_topk(output, target, 5)
            print(f"[Epoch {epoch+1} | validated Batch {i+1}/{total_batches}]")
            print(f"[Epoch {epoch+1} | validated Batch {i+1}/{total_batches}]", file = logfile)
    losses_m /= total
    acc_top1 = correct_top1 / total
    acc_top5 = correct_top5 / total

    metrics = OrderedDict([('loss', float(losses_m)),
                           ('acc_top1', float(acc_top1)),
                           ('acc_top5', float(acc_top5))])

    return metrics
def main(args):
    
    logfile = open("log_torch.txt", "w")
    def custom_normalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) 
        return (tensor - mean) / std
    
    train_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = True,
        transform = Compose([
            Resize(64),
            RandomHorizontalFlip(),  
            ToTensor(),
            custom_normalize
        ]),
        debug = True
    )
    val_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = False,
        transform = Compose([
            Resize(64),
            ToTensor(),
            custom_normalize
        ]) 
    )
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True) 
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
    load_model = False
    MODEL_PATH = "overlock-torch.pt"
    model = models_torch.overlock_xxt(num_classes = TinyImageNet.N_CLASS)
    if load_model:
        model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3) 
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss()
    start_epoch = 0
    num_epochs = 20
    
    train_loss = []
    train_loss_aux = []
    epoch_time = []
    val_loss = []
    acc_top1 = []
    acc_top5 = []
    for epoch in range(start_epoch, num_epochs):

        train_metrics = train_one_epoch(
            epoch, model, train_loader, 
            optimizer, train_loss_fn, args, logfile
        )
        train_loss.append(train_metrics['loss'])
        train_loss_aux.append(train_metrics['loss_aux'])
        epoch_time.append(train_metrics['epoch_time'])
        
        
        val_metrics = validate(epoch, model, val_loader, validate_loss_fn, args, logfile)
        val_loss.append(val_metrics["loss"])
        acc_top1.append(val_metrics["acc_top1"])
        acc_top5.append(val_metrics["acc_top5"])
        print(f"[Epoch {epoch+1}")
        print(f"[Epoch {epoch+1}", file = logfile)
        for key, value in train_metrics.items():
            print(f"{key}: {value}")
            print(f"{key}: {value}", file = logfile)
        for key, value in val_metrics.items():
            print(f"{key}: {value}")
            print(f"{key}: {value}", file = logfile)
        print("]")
        print("]", file = logfile)
    log = {
        "train_loss":train_loss, 
        "train_loss_aux":train_loss_aux,
        "epoch_time":epoch_time,
        "val_loss":val_loss,
        "acc_top1":acc_top1,
        "acc_top5":acc_top5
    }
    pd.DataFrame(log).to_csv("log_torch.csv", index = False, header = True)
    torch.save(model.state_dict(), MODEL_PATH)
    logfile.close()
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
```



## 结果和分析

使用`plot.py`程序根据两个版本的测试结果（`log.csv`和`log_torch.csv`）绘制训练时损失函数和分类精度的变化曲线。

![training_metrics](training_metrics.png)

训练损失由主损失和辅助损失（均为交叉熵损失）相加得到，L = L(main) + α * L(aux)，其中根据OverLoCK原版实现取α=0.4。

在图中可看到，辅助损失（Auxiliary Loss）在两个版本的实现中的变化是高度一致的，这是因为它是经过OverLoCK模型中Base-net和Overview-net计算得到的结果，这两个子网络中模块的Jittor实现与PyTorch实现完全相同。

然而主损失的两个版本的实现差异较大。在第二张图（Main Loss）中，实线代表两个版本在训练时的主损失，虚线代表验证集上计算得到的损失（根据原版OverLoCK实现，验证时仅以Focus-net的输出计算主损失，无辅助损失）。从epoch 10开始，PyTorch版本的训练验证损失与训练主损失差距显著增大，造成明显的过拟合；Jittor版本则尚未出现过拟合，但损失下降速度明显比PyTorch版本更慢。由于时间限制，没有继续训练，但推测其损失会继续下降。

分析：问题的关键在于Dynamic Block中使用的`na2d_av`算子的复现不够精确。OverLoCK模型中使用的`na2d_av`算子源自[Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer)的研究。原版OverLoCK实现使用的`na2d_av`算子来自第三方库NATTEN，它在NAT的研究基础上基于C++和CUDA做了高性能的底层实现。但由于实现过于复杂，本实验在编写Jittor版本时只根据NAT论文中的说明实现了朴素的邻域注意力计算，这可能使得计算原理上与NATTEN差异较大，导致表示能力弱于原版；即使有部分等价的计算效果，也可能因为与高度优化的计算图不一致而造成训练过程中优化问题地形的不同，收敛速度更慢。

后两张图是验证集上的Top-1和Top-5分类精度（百分数）。由于上述复现的差异，在训练过程的前半段，精度的差异较大；但是随着epoch的增大，两个版本的精度变得非常接近，这说明在给定的数据集和模型设置条件下，模型最终学习到的表示质量和泛化性能是一致的。最终的精度情况为：

|         | acc@1 | acc@5 |
| ------- | ----- | ----- |
| Jittor  | 0.469 | 0.792 |
| PyTorch | 0.485 | 0.819 |

Jittor训练总时长为259.2 min，PyTorch训练总时长为117.2 min，前者是后者的两倍多。除了计算框架本身的差异，造成Jittor训练缓慢可能的原因还有上述`na2d_av`算子的朴素实现造成更大的计算开销。

详细的训练日志见`log.txt`和`log_torch.txt`。

