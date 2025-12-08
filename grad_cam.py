
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import numpy as np
import os
from PIL import Image 
import models
from jittor.dataset import DataLoader, Dataset
from jittor.transform import Compose, Resize, RandomCrop, CenterCrop, ToTensor, RandomHorizontalFlip
from jittor import nn
import jittor
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model):

        self.gradients = []
        self.model = model
        for name, param in model.named_parameters():
            param.requires_grad = True

        
        self.model.blocks1[-1].register_forward_hook(self.save_feature_basenet)
        self.model.blocks2[-1].register_forward_hook(self.save_feature_basenet)
        self.model.sub_blocks3[-1].register_forward_hook(self.save_feature_focusnet)
        self.model.sub_blocks4[-1].register_forward_hook(self.save_feature_focusnet)
        
        self.model.blocks1[-1].register_backward_hook(self.save_gradient_basenet)
        self.model.blocks2[-1].register_backward_hook(self.save_gradient_basenet)
        self.model.sub_blocks3[-1].register_backward_hook(self.save_gradient_focusnet)
        self.model.sub_blocks4[-1].register_backward_hook(self.save_gradient_focusnet)

        
        
        
        self.features = []
    def save_gradient_focusnet(self, module, grad_input, grad_output):

        self.gradients.append(grad_output[0])
    def save_gradient_basenet(self, module, grad_input, grad_output):

        self.gradients.append(grad_output[0])
    def save_feature_focusnet(self, module, input, output):

        self.features.append(output[0]) # 只保留特征图，不保留上下文先验
    def save_feature_basenet(self, module, input, output):

        self.features.append(output)
    def __call__(self, x):
        x = self.model(x)
        return self.features, x


class ModelOutputs():
    def __init__(self, model):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model)
    def get_gradients(self):
        return list(reversed(self.feature_extractor.gradients)) 
    def __call__(self, x):
        features, output = self.feature_extractor(x)  
        return features, output

class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = ModelOutputs(self.model)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        features, output = self.extractor(input)
        
        predict, _ = jittor.argmax(output, dim = -1)

        if index is None:
            index = predict


        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = jittor.var(one_hot) 

        loss = jittor.sum(one_hot * output)
        
        grads = jittor.grad(loss, self.model.parameters())
        grads_val = self.extractor.get_gradients()
        cams = []
        print(len(features), len(grads_val))
        for target, grad in zip(features, grads_val):
            target = target.numpy()[0]
            grad = grad.numpy()[0]

            weight = np.mean(grad, axis = (1, 2))
            cam = np.zeros(target.shape[1:], dtype = np.float32)
            for i, w in enumerate(weight):
                cam += w * target[i, :, :]
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, (64, 64))
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
            cams.append(cam)
        return cams, predict

        

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
            img_ = self.transform(img)
        else:
            img_ = img
        return img_, jittor.int32(label), img
if __name__ == '__main__':

    def custom_normalize(tensor):
        mean = jittor.array([0.485, 0.456, 0.406]).view(3,1,1)
        std = jittor.array([0.229, 0.224, 0.225]).view(3,1,1) 
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
    MODEL_PATH = "overlock.pkl"
    model = models.overlock_xxt(num_classes = TinyImageNet.N_CLASS)
    
    model.load_state_dict(jittor.load(MODEL_PATH))

    for ind in range(20):
        ind = ind * 50 ## 对各个类别
        input, target, img = val_dataset[ind]
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
        
        grad_cam = GradCam(model)
               
        cams, predict = grad_cam(input, target) # 检测groundtruth类别的激活图

        correct = (predict.item() == target.item())

        plt.figure(figsize = (30, 10))
        n = len(cams) + 1
        plt.subplot(1, n, 1)
        plt.imshow(img)
        plt.axis('off')
        for i, cam in enumerate(cams):
            heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            cam = heatmap + np.float32(img) / 255
            cam = cam / np.max(cam)
            plt.subplot(1, n, i + 2)
            plt.imshow(cam)
            plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(f'visualization/jittor_cam_{ind}_{correct}.png', bbox_inches = 'tight')