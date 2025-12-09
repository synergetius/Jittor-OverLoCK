import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, ToTensor, RandomHorizontalFlip
import os
from PIL import Image 
import models_torch
device = torch.device("cuda")

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
            img_ = self.transform(img)
        else:
            img_ = img
        return img_.to(device), torch.tensor(label, dtype = torch.int64).to(device), img
        

MODEL_PATH = "overlock-torch.pt"
model = models_torch.overlock_xxt(num_classes = TinyImageNet.N_CLASS)
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(device)
#定义输入图像的长宽，这里需要保证每张图像都要相同
#生成一个和输入图像大小相同的0矩阵，用于更新梯度
heatmap = np.zeros([64, 64])
def custom_normalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) 
    return (tensor - mean) / std
val_dataset = TinyImageNet(
    root = '/mnt/d/data/tiny-imagenet-200',
    train = False,
    transform = Compose([
        Resize(64),
        ToTensor(),
        custom_normalize
    ]) 
)

heatmaps = [np.zeros([64, 64]) for i in range(4)]
features = []
def save_feature_focusnet(module, input, output):
    features.append(output[0])
def save_feature_basenet(module, input, output):
    features.append(output)

model.blocks1[-1].register_forward_hook(save_feature_basenet)
model.blocks2[-1].register_forward_hook(save_feature_basenet)
model.sub_blocks3[-1].register_forward_hook(save_feature_focusnet)
model.sub_blocks4[-1].register_forward_hook(save_feature_focusnet)
for ind in range(200):
    ind = ind * 5 # 200张图像，分散采样
    print(ind)
    input, target, img = val_dataset[ind]
    input = input.unsqueeze(0)
    target = target.unsqueeze(0)
    input.requires_grad = True
    features.clear()
    
    output = model(input)
    for i, feature in enumerate(features):
        feature = feature.mean(dim = 1,keepdim = False).squeeze()
        model.zero_grad()
        feature[feature.shape[0]//2-1][feature.shape[1]//2-1].backward(retain_graph=True)
        grad = torch.abs(input.grad)
        grad = grad.mean(dim=1,keepdim=False).squeeze()
        heatmaps[i] = heatmaps[i] + grad.cpu().numpy()

#对累加的梯度进行归一化
heatmap = heatmap - heatmap.min()
heatmap = heatmap / heatmap.max()

#可视化，蓝色值小，红色值大
heatmap = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
plt.figure(figsize = (30, 10))
for i, heatmap in enumerate(heatmaps):
    plt.subplot(1, len(heatmaps), i + 1)
    plt.imshow(heatmap)
    plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.savefig(f'visualization/torch_erf.png', bbox_inches = 'tight')