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
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model):
        self.model = model
        self.gradients = []
        self.features = []
    def save_gradient_focusnet(self, module, grad_input, grad_output):
        # print("grad_output:", type(grad_output), len(grad_output))
        self.gradients.append(grad_output[0]) # 只保留类别分数对特征图的梯度，不保留对上下文先验的梯度
    def save_gradient_basenet(self, module, grad_input, grad_output):
        # print("!", type(grad_output))
        self.gradients.append(grad_output[0])
    def save_feature_focusnet(self, module, input, output):
        # print("output:", type(output), len(output))
        self.features.append(output[0]) # 只保留特征图，不保留上下文先验
    def save_feature_basenet(self, module, input, output):
        # print("!!", type(output))
        self.features.append(output)
    def __call__(self, x):
        self.gradients = []
        self.model.blocks1[-1].register_backward_hook(self.save_gradient_basenet)
        self.model.blocks2[-1].register_backward_hook(self.save_gradient_basenet)
        self.model.sub_blocks3[-1].register_backward_hook(self.save_gradient_focusnet)
        self.model.sub_blocks4[-1].register_backward_hook(self.save_gradient_focusnet)
        
        self.model.blocks1[-1].register_forward_hook(self.save_feature_basenet)
        self.model.blocks2[-1].register_forward_hook(self.save_feature_basenet)
        self.model.sub_blocks3[-1].register_forward_hook(self.save_feature_focusnet)
        self.model.sub_blocks4[-1].register_forward_hook(self.save_feature_focusnet)
        
        
        x = self.model(x)
        return self.features, x
        # for name, module in self.model._modules.items():
            # x = module(x)
            # if name in self.target_layers:
                # x.register_hook(self.save_gradient)
                # outputs += [x]
        # return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model)

    def get_gradients(self):
        return list(reversed(self.feature_extractor.gradients)) 
        # 对特征图的梯度是从后到前添加的，所以倒序后返回

    def __call__(self, x):
        features, output = self.feature_extractor(x)
        # print("features:", features[0].shape, features[1].shape)
        # print("output:", output.shape)
        
        # features: torch.Size([1, 96, 4, 4]) torch.Size([1, 160, 2, 2])
        # output: torch.Size([1, 20])     

        return features, output
        # print(features, output)
        # print(features[0][0].shape, features[0][1].shape)#, len(features[0]))
        # print(features[0].shape, features[1].shape)
        # print(output.shape)
        # target_activations, output  = self.feature_extractor(x)
        # output = output.view(output.size(0), -1)
        # output = self.model.classifier(output)
        # return target_activations, output

# def show_cam_on_image(img, mask):
    # heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    # heatmap = np.float32(heatmap) / 255
    # cam = heatmap + np.float32(img)
    # cam = cam / np.max(cam)
    # cv2.imwrite("../../images/cam01.jpg", np.uint8(255 * cam))

class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = ModelOutputs(self.model)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        features, output = self.extractor(input)
        predict = torch.argmax(output)
        if index is None:
            index = predict
            # index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True).to(device)
        # print(one_hot.shape, output.shape)
        one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        #one_hot.backward(retain_variables=True)
        one_hot.backward()
        grads_val = self.extractor.get_gradients()#[-1].cpu().data.numpy()
        # print("grads_val:", grads_val[0].shape, grads_val[1].shape)
        cams = []
        for target, grad in zip(features, grads_val):
            target = target.cpu().data.numpy()[0]
            grad = grad.cpu().data.numpy()[0]
            # print(target.shape, grad.shape)
            weight = np.mean(grad, axis = (1, 2))
            # print(weight.shape)
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
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
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
    # val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False)
    MODEL_PATH = "overlock-torch.pt"
    model = models_torch.overlock_xxt(num_classes = TinyImageNet.N_CLASS)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(device)
    
    # for input, target in val_loader:
        # print('input shape:', input.shape)
        # break
    for ind in range(20):
        ind = ind * 50 ## 对各个类别
        input, target, img = val_dataset[ind]
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)
        # print('input shape:', input.shape)
        
        grad_cam = GradCam(model)
                        
        cams, predict = grad_cam(input, target) # 检测groundtruth类别的激活图
        # cams, index = grad_cam(input)
        correct = (predict.item() == target.item())
        
        # for cam in cams:
            # print(cam.shape)
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
        plt.savefig(f'visualization/torch_cam_{ind}_{correct}.png', bbox_inches = 'tight')
    # show_cam_on_image(input, cams)
    """
    image_path = "../../images/dog-cat.jpg"

    # Can work with any model, but it assumes that the model has a 
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model = models.vgg19(pretrained=True), \
                    target_layer_names = ["35"], use_cuda=True)

    img = cv2.imread(image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None

    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model = models.vgg19(pretrained=True), use_cuda=True)
    gb = gb_model(input, index=target_index)
    utils.save_image(torch.from_numpy(gb), '../../images/gb.jpg')

    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    utils.save_image(torch.from_numpy(cam_gb), '../../images/cam_gb.jpg')
    """