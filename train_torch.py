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
                        help='input batch size for training (default: 64)') ############### 暂时修改，测试存储分配问题
    parser.add_argument('--aux-loss-ratio', type=float, default=0.4,
                        help='Aux loss weight')
    return parser
class TinyImageNet(Dataset):
    N_CLASS = 20 # 选择的类别数（按编号字典序）
    N_TRAIN = 500 # 每个类别的训练样本数（TinyImageNet数据集每个类别总共有500个）
    N_VAL = 50 # 每个类别的验证样本数（TinyImageNet总共只有50个）
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
                #print(f"[DEBUG] 开始扫描类别 {cls} ")  
                ls = sorted(os.listdir(cls_dir)) 
                ls = ls[:self.N_TRAIN]# 固定顺序选取一部分
                sample_count[i] += len(ls)
                for img_name in ls: ######## 尝试每个类别只用一小部分训练，降低计算量
                    self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
            plot = True
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('train_samples.png', dpi=300, bbox_inches='tight')
                # 200个类别样本数量是均匀的
                
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")  
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            #lines = lines[:len(lines) // 8] ####### 只取（按注释文件顺序的）一小部分作为验证集
            # self.img_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in lines}
            self.class_to_img = {}
            for line in lines:
                img, cls = line.split('\t')[:2]
                self.class_to_img.setdefault(cls, []).append(img)
            
            self.classes = sorted(list(self.class_to_img.keys()))[:self.N_CLASS] ##########
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            # 200 classes, each 50 samples
            print('validation classes number:', len(self.classes))
            sample_count = np.zeros(len(self.classes))
            
            
            self.samples = []
            for cls in self.classes:
            #for cls, imgs in self.class_to_img.items():
                imgs = self.class_to_img[cls]
                # 每个类共50个样本，按注释文件顺序，取前面的一部分
                for img in imgs[:self.N_VAL]:
                    sample_count[self.class_to_idx[cls]] += 1
                    self.samples.append((os.path.join(self.data_dir, img), self.class_to_idx[cls]))
            plot = True
            if plot:
                plt.bar(self.classes, sample_count)
                plt.savefig('val_samples.png', dpi=300, bbox_inches='tight')
            # self.samples = [
                # (os.path.join(self.data_dir, img_name), self.class_to_idx[self.img_to_class[img_name]])
                # for img_name in self.img_to_class.keys()
                # #if os.path.isfile(os.path.join(self.data_dir, img_name))
            # ]
        #self.set_attrs(total_len=len(self.samples), batch_size=128, shuffle=True)

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # return img, label
        return img.to(device), torch.tensor(label, dtype = torch.int64).to(device)
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, logfile):
    model.train()
    total_batches = len(loader)
    time0 = time.time()
    losses_m = 0
    losses_aux_m = 0
    total = 0 # 样本总数
    for i, (input, target) in enumerate(loader):
        # if i >= 1:
            # break
        try:
            optimizer.zero_grad() # 清零梯度
            
            output = model(input)
            output_main = output['main']
            output_aux = output['aux']
            loss_main = loss_fn(output_main, target)
            loss_aux = loss_fn(output_aux, target)
            loss = loss_main + args.aux_loss_ratio * loss_aux
            loss.backward()
            
            losses_m += loss * input.size(0) #loss_fn默认对batch内所有样本取平均
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
    # print(predicted, target)
    correct = (predicted == target.unsqueeze(-1)).sum().float().item()
    return correct
def validate(epoch, model, loader, loss_fn, args, logfile):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_batches = len(loader)
    # print(total_batches)
    losses_m = 0
    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            # if i >= 1: #### for test
                # break
            output = model(input)
            #print(target)
            tgt = F.one_hot(target, num_classes = TinyImageNet.N_CLASS).to(torch.float)
            #print(target)
            #print(output.shape, target.shape)
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
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) ###### 这里执行通道维度转换
        return (tensor - mean) / std
    
    train_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = True,
        transform = Compose([
            Resize(64),
            #RandomCrop(224),
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
            #CenterCrop(224),
            ToTensor(),
            custom_normalize
        ]) ######### 需要一致
    )
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True) 
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
    load_model = False
    MODEL_PATH = "overlock-torch.pt"
    model = models_torch.overlock_xxt(num_classes = TinyImageNet.N_CLASS)
    if load_model:
        model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3) ####### 仅作为测试
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss()
    ############# test
    start_epoch = 0 ########
    num_epochs = 20 ########
    # num_epochs = 1 ########
    
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
    # log = pd.DataFrame({
        # "train_loss":train_loss, 
        # "train_loss_aux":train_loss_aux,
        # "val_loss":val_loss,
        # "acc_top1":acc_top1,
        # "acc_top5":acc_top5
    # })
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