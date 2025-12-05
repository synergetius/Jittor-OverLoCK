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
                        help='input batch size for training (default: 128)') ############### 暂时修改，测试存储分配问题
    parser.add_argument('--aux-loss-ratio', type=float, default=0.4,
                        help='Aux loss weight')
    return parser
class TinyImageNet(Dataset):
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
            self.classes = sorted(os.listdir(self.data_dir))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            print(f"[DEBUG] 找到 {len(self.classes)} 个类别")  
            sample_ind = os.path.join(self.root, 'train_samples.csv')
            # if os.path.isfile(sample_ind):
                # with open(sample_ind, mode = "r") as f:
                    # self.samples = [(path, int(cls_idx)) for path, cls_idx in csv.reader(f)]
            # else:
            print('classes number:', len(self.classes))
            
            sample_count = np.zeros(len(self.classes))

            for i, cls in enumerate(self.classes):
                cls_dir = os.path.join(self.data_dir, cls, 'images')
                cls_idx = self.class_to_idx[cls]
                #print(f"[DEBUG] 开始扫描类别 {cls} ")  
                ls = sorted(os.listdir(cls_dir)) 
                ls = ls[:len(ls) // 8]# 固定顺序选取一部分
                sample_count[i] += len(ls)
                for img_name in ls: ######## 尝试每个类别只用一小部分训练，降低计算量
                    self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
            with open(sample_ind, mode = "w") as f:
                csv.writer(f).writerows(self.samples)
            plot = False
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
            self.classes = sorted(list(self.class_to_img.keys()))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            # 200 classes, each 50 samples
            print('validation classes number:', len(self.classes))
            sample_count = np.zeros(len(self.classes))
            
            
            self.samples = []
            for cls, imgs in self.class_to_img.items():
                # 每个类共50个样本，取按注释文件顺序的前12个样本，共2400个
                for img in imgs[:12]:
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
        self.set_attrs(total_len=len(self.samples))
        #self.set_attrs(total_len=len(self.samples), batch_size=128, shuffle=True)


    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # return img, label
        return img, jittor.int32(label)
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args, logfile):
    model.train()
    total_batches = len(loader)
    time0 = time.time()
    losses_m = 0
    losses_aux_m = 0
    total = 0 # 样本总数
    for i, (input, target) in enumerate(loader):
        # if i >= 0:
        # if epoch == 0:
            
            # break ############### for test
     
        try:
            output = model(input)
            output_main = output['main']
            output_aux = output['aux']
            loss_main = loss_fn(output_main, target)
            loss_aux = loss_fn(output_aux, target)
            loss = loss_main + args.aux_loss_ratio * loss_aux
            losses_m += loss * input.size(0) #loss_fn默认对batch内所有样本取平均
            losses_aux_m += loss_aux * input.size(0)
            optimizer.step(loss)
            #print(loss)
            #print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}]")
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
    
    predicted, _ = output.topk(k, dim = 1)
    correct = (predicted == target).sum().float().item()
    return correct
def validate(epoch, model, loader, loss_fn, args, logfile):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_batches = len(loader)
    # print(total_batches)
    losses_m = 0
    total = 0
    with jittor.no_grad():
        for i, (input, target) in enumerate(loader):
            # if i >= 1: #### for test
                # print(i)
                # continue
            output = model(input)
            # print(output.shape)
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
    
    # os.environ['JT_SYNC'] = '1'
    # os.environ['trace_py_var'] = '3' 
    # os.environ['JT_LOG'] = '1'
    # os.environ['JT_SAVE_MEM'] = '1'
    # os.environ['JT_OPT_LEVEL'] = '0'
    jittor.flags.use_cuda = 1
    jittor.flags.log_silent = 0
    jittor.flags.log_v = 1
    logfile = open("log.txt", "w")
    #jittor.flags.lazy_execution = 0
    def custom_normalize(tensor):
        mean = jittor.array([0.485, 0.456, 0.406]).view(3,1,1)
        std = jittor.array([0.229, 0.224, 0.225]).view(3,1,1) ###### 这里执行通道维度转换
        return (tensor - mean) / std
    
    train_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = True,
        transform = Compose([
            Resize(256),
            RandomCrop(224),
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
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            custom_normalize
        ]) ######### 需要一致
    )
    train_loader = train_dataset.set_attrs(batch_size = args.batch_size, shuffle = True) 
    val_loader = val_dataset.set_attrs(batch_size = args.batch_size, shuffle=False) #num_workers 有何作用？？？
    #model = models.overlock_xt()
    model = models.overlock_xxt() ############
    
    optimizer = jittor.optim.AdamW(model.parameters(), lr = 1e-3) ####### 仅作为测试
    train_loss_fn = nn.CrossEntropyLoss()
    validate_loss_fn = nn.CrossEntropyLoss()
    ############# test
    start_epoch = 0
    num_epochs = 10
    
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
    pd.DataFrame(log).to_csv("log.csv", index = False, header = True)
    jittor.save(model.state_dict(), "overlock.pkl")
    logfile.close()
    # log.to_csv('log.csv', index = False, header = True)
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
    
    