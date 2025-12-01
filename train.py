import argparse
import datetime
import json
import csv
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress

from jittor.dataset import DataLoader, Dataset
from jittor.transform import Compose, Resize, RandomCrop, ToTensor, RandomHorizontalFlip
from jittor import nn
import jittor
import models
from PIL import Image  
def get_args_parser():
    parser = argparse.ArgumentParser(description='Jittor Training', add_help=False)
    parser.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
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
            if os.path.isfile(sample_ind):
                with open(sample_ind, mode = "r") as f:
                    self.samples = [(path, int(cls_idx)) for path, cls_idx in csv.reader(f)]
            else:
                
                for cls in self.classes:
                    cls_dir = os.path.join(self.data_dir, cls, 'images')
                    cls_idx = self.class_to_idx[cls]
                    print(f"[DEBUG] 开始扫描类别 {cls} ")  
                    for img_name in os.listdir(cls_dir):
                        self.samples.append((os.path.join(cls_dir, img_name), cls_idx))
                with open(sample_ind, mode = "w") as f:
                    csv.writer(f).writerows(self.samples)
        else:
            print(f"[DEBUG] 开始读取验证集注释文件: {self.annotation_file}")  
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            self.img_to_class = {line.split('\t')[0]: line.split('\t')[1] for line in lines}
            self.classes = sorted(list(set(self.img_to_class.values())))
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
            self.samples = [
                (os.path.join(self.data_dir, img_name), self.class_to_idx[self.img_to_class[img_name]])
                for img_name in self.img_to_class.keys()
                if os.path.isfile(os.path.join(self.data_dir, img_name))
            ]
        self.set_attrs(total_len=len(self.samples))
        #self.set_attrs(total_len=len(self.samples), batch_size=128, shuffle=True)


    def __getitem__(self, idx):
        
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # return img, label
        return img, jittor.int32(label)
def train_one_epoch(epoch, model, loader, optimizer, loss_fn, args):
    for i, (inputs, targets) in enumerate(loader):
        # if epoch == start_epoch and i < start_batch:
            # continue

        try:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            optimizer.step(loss)

            print(f"[Epoch {epoch+1} | Batch {i+1}/{total_batches}] Loss: {loss.item():.4f}")


        except Exception as e:
            print(f"训练第{i+1}个 batch 时出错: {e}")
            raise
def main(args):
    def custom_normalize(tensor):
        mean = jittor.array([0.485, 0.456, 0.406]).view(3,1,1)
        std = jittor.array([0.229, 0.224, 0.225]).view(3,1,1) ###### 这里执行通道维度转换
        return (tensor - mean) / std
    transform = Compose([
        Resize(256),
        RandomCrop(224),
        RandomHorizontalFlip(),  
        ToTensor(),
        custom_normalize
    ])
    train_dataset = TinyImageNet(
        root = '/mnt/d/data/tiny-imagenet-200',
        train = True,
        transform = transform,
        debug = True
    )
    train_loader = train_dataset.set_attrs(batch_size = args.batch_size, shuffle = True) 
    model = models.overlock_xt()
    optimizer = jittor.optim.AdamW(model.parameters(), lr = 1e-3) ####### 仅作为测试
    train_loss_fn = nn.CrossEntropyLoss()
    ############# test
    start_epoch = 0
    num_epochs = 1
    for epoch in range(start_epoch, num_epochs):
        # if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
            # loader_train.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            epoch, model, train_loader, 
            optimizer, train_loss_fn, args,
            #lr_scheduler=lr_scheduler, 
            #saver=saver, output_dir=output_dir,
            #amp_autocast=amp_autocast, 
            #loss_scaler=loss_scaler, 
            #model_ema=model_ema, 
            #mixup_fn=mixup_fn,
            #eta_meter=AverageMeter()
        )
        if (epoch % args.val_freq == 0) or epoch > args.val_start_epoch or resume_mode:
            resume_mode = False
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                distribute_bn(model, args.world_size,
                              args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, 
                                    args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    distribute_bn(model_ema, args.world_size,
                                  args.dist_bn == 'reduce')
                ema_eval_metrics = validate(model_ema.module, loader_eval,
                                            validate_loss_fn, args,
                                            amp_autocast=amp_autocast,
                                            log_suffix=f'_{ema_save_tag}')
                eval_metrics.update(ema_eval_metrics)
        else:
            eval_metrics = {key: float('-inf') for key in eval_metrics}

        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        if output_dir is not None:
            update_summary(epoch,
                           train_metrics,
                           eval_metrics,
                           os.path.join(output_dir, 'summary.csv'),
                           write_header=best_metric is None,
                           log_wandb=args.log_wandb and has_wandb)

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

        if saver_ema is not None:
            # save proper checkpoint with eval metric (ema)
            save_metric = eval_metrics[eval_metric_ema]
            best_metric_ema, best_epoch_ema = saver_ema.save_checkpoint(epoch, metric=save_metric)

        
        if best_metric is not None:
            
            if args.local_rank == 0:
                _logger.info(f"Currently Best Accuracy: {best_metric:.2f} at Epoch {best_epoch}")
                if best_metric_ema is not None:
                    _logger.info(f"Currently Best Accuracy (EMA): {best_metric_ema:.2f} at Epoch {best_epoch_ema}")
                _logger.info('\n')
                
            with open(os.path.join(output_dir, 'best-metric.json'), 'w') as f:
                
                best_metric_info = {
                    eval_metric: best_metric,
                    'epoch': best_epoch,
                }
                
                if best_metric_ema is not None:
                    best_metric_info[eval_metric_ema] = best_metric_ema
                    best_metric_info['epoch_ema'] = best_epoch_ema
                    
                json.dump(best_metric_info, f, indent=4)
if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)