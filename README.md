# Jittor-OverLoCK

本项目基于Jittor框架，对CVPR 2025的论文[OverLoCK](https://github.com/LMMMEng/OverLoCK)进行预训练环节的实验复现，并编写PyTorch版本的程序进行对齐实验。

## 环境配置

在Windows系统中，由于兼容性问题安装Jittor失败，所以安装了WSL Ubuntu 22.04.5用于实验。

安装了anaconda 3用于管理python环境。

Jittor的训练环境使用python 3.7，使用官网提供的安装命令：

```
python -m pip install jittor
python -m jittor.test.test_example
python -m jittor.test.test_cudnn_op
```

PyTorch的训练环境根据OverLoCK官方版本的实现配置：

```
# Environments:
cuda==12.1
python==3.10
# Dependencies:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install natten==0.17.1+torch230cu121 -f https://shi-labs.com/natten/wheels/
pip install timm==0.6.12
pip install mmengine==0.2.0
```

## 数据集

采用TinyImagenet数据集，它包含200个类别，训练集每个类别有500张图像，验证集每个类别有50张图像，尺寸均为64x64。由于计算资源和时间的限制，将类别序号按字典序排序后选出前20个类别进行训练和测试。

## 训练和测试

`train.py`是Jittor的训练和测试脚本，包含数据集加载、训练过程和验证集上的评估。

`train_torch.py`是PyTorch的训练和测试脚本，实现基本相同的训练和测试流程。

## 结果

使用`plot.py`程序根据两个版本的测试结果（`log.csv`和`log_torch.csv`）绘制训练时损失函数和分类精度的变化曲线。

详细的训练日志见`log.txt`和`log_torch.txt`。

