This repository provides code for paper 《Lightweight Deep Neural Network Model With Padding-free Downsampling》


For the CIFAR-100 dataset, using the first set of hyperparameters
--
The first set of hyperparameters follows the settings of Haase et al. 
<br>

__《Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets》<br>__
Daniel Haase∗ &nbsp; Manuel Amthor∗ <br>
ZEISS Microscopy  &nbsp; ZEISS Microscopy

## Orig——CIFAR-100
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 3.066M | 68.5M |  75.37% |  
| MobileNetv3-large(BSConv-S) | 3.066M | 68.5M |  77.87% |  
| ResNet-20 | 0.278M | 41.4M |  68.12% |  
| ResNet-110(BSConv-U) | 0.245M | 41.8M |  71.58% |  
| WideResNet-40-3 | 5.056M | 735.8M |  76.23% |  
| WideResNet-40-8(BSConv-U) | 4.286M | 675.1M |  77.79% |  

## Ours——CIFAR-100
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 3.067M | 54.6M ↓|  75.71% |  
| MobileNetv3-large(BSConv-S) | 3.067M | 54.6M ↓|  78.36% |  
| ResNet-20 | 0.282M | 37.8M ↓|  68.30% |  
| ResNet-110(BSConv-U) | 0.249M | 38.6M ↓|  71.62% |  
| WideResNet-40-3 | 5.287M | 668.7M ↓|  76.28% |  
| WideResNet-40-8(BSConv-U) | 4.457M | 615.6M ↓|  78.05% |  

<br>

For the CIFAR-100 dataset, using the second set of hyperparameters
--
## Orig——CIFAR-100
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 4.330M | 68.8M |  76.00% |  
| Parc-MobileNet-v2 | 2.348M | 91.3M |  76.20% |  
| GhostNet | 4.029M | 44.6M |  74.00% |  
| ShuffleNet-v2 | 1.356M | 46.2M |  70.90% |  

## Ours——CIFAR-100
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 4.331M | 54.7M ↓|  76.60% |  
| Parc-MobileNet-v2 | 2.348M | 73.0M ↓|  76.60% |  
| GhostNet | 4.030M | 34.8M ↓|  74.10% |  
| ShuffleNet-v2 | 1.358M | 35.7M ↓|  71.50% |  

<br>

For Stanford Dogs dataset
--

## Orig——Stanford Dogs
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 3.086M | 230.1M |  51.07% |  
| MobileNetv3-large-bsconvs | 3.086M | 230.1M |  59.68% |  

## Ours——Stanford Dogs
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 3.087M | 212.6M ↓|  54.11% |  
| MobileNetv3-large-bsconvs | 3.087M | 212.6M ↓|  60.79% |  

<br>

For ImageNet dataset
--

## Orig——ImageNet
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 5.480M | 232.5M |  69.50% |  

## Ours——ImageNet
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 5.481M | 214.9M ↓|  69.50% |  

<br>

For VegFru-292 dataset
--
## Orig——VegFru-292
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 4.576M | 224.5M |  89.20% |  
| Parc-MobileNet-v2 | 2.605M | 314.8M |  89.1% |  
| GhostNet | 4.276M | 147.9M |  89.60% |  
| ShuffleNet-v2 | 1.553M | 148.1M |  88.40% |  

## Ours——VegFru-292
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 4.577M | 205.7M ↓|  89.90% |  
| Parc-MobileNet-v2 | 2.605M | 305.5M ↓|  90.0% |  
| GhostNet | 4.276M | 136.9M ↓|  90.30% |  
| ShuffleNet-v2 | 1.554M | 130.7M ↓|  87.70% |  

<br>

Inference Latency
--
## Orig——Latency
| Model | AMD Ryzen 5 5600H | MediaTek Tiangui 1000+ |
| :---: | :---: | :---: | 
| MobileNetv3-large | 8.5ms | 27.0ms |
| Parc-MobileNet-v2 | 8.7ms | 37.4ms |
| GhostNet | 11.4ms | 36.6ms |
| ShuffleNet-v2 | 6.2ms | 19.4ms |

## Ours——Latency
| Model | AMD Ryzen 5 5600H | MediaTek Tiangui 1000+ |
| :---: | :---: | :---: | 
| MobileNetv3-large | 9.0ms | 26.3ms ↓|
| Parc-MobileNet-v2 | 9.3ms | 34.0ms ↓|
| GhostNet | 11.7ms | 26.8ms ↓|
| ShuffleNet-v2 | 7.4ms | 18.8ms ↓|

<br>

Ablation Experiments on CIFAR-100
--
| Model | orig | +Stem |  +Downsampling | ours |
| :---: | :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 76.0%  | 75.9% |  76.4% | 76.6%↑ |  
| Parc-MobileNet-v2 | 76.2%  | 76.6% |  76.4% | 76.6%↑ |  
| GhostNet | 76.0%  | 74.2% |  73.8% | 74.1%↑ |  
| ShuffleNet-v2 | 70.9% | 72.0% | 70.4% | 71.5%↑ |  

<br>

Comparison with other downsampling (EfficientFormerv2)
--
__《Rethinking Vision Transformers for MobileNet Size and Speed》<br>__
Yanyu Li <br>
Snap Inc. Northeastern University 

## EfficientFormerv2-Downsampling——CIFAR-100
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 4.317M | 78.0M | 75.80% |  
| Parc-MobileNet-v2 | 2.558M | 97.5M |  75.70% |  
| GhostNet | 4.092M | 58.3M |  74.30% |  
| ShuffleNet-v2 | 2.804M | 84.1M |  70.60% | 

## Ours——CIFAR-100
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 4.331M | 54.7M |  76.60% |  
| Parc-MobileNet-v2 | 2.348M | 73.0M |  76.60% |  
| GhostNet | 4.030M | 34.8M |  74.10% |  
| ShuffleNet-v2 | 1.358M | 35.7M |  71.50% | 

Getting started
--

* For the BSConv folder, when using it for the first time, use "--download" to download the dataset.
```python
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_mobilenetv3_large_w1 --download --gpu-id 0
```

* "--data-root" is the dataset path, "--dataset" is the dataset name, "--architecture" is the model name.

```python
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_mobilenetv3_large_w1 --gpu-id 0
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_mobilenetv3_large_w1_bsconvs_p1d6 --gpu-id 0
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_wrn40_3 --gpu-id 0
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_wrn40_8_bsconvu --gpu-id 0
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_resnet20 --gpu-id 0
python bsconv_pytorch_train.py --data-root cifar100 --dataset cifar100 --architecture cifar_resnet110_bsconvu --gpu-id 0 
```
Take MobileNetv3 as an example, when using our module.
--

* Replace init_conv on line 321 in the mobilenet.py file with our stem layer.
```python
self.backbone.add_module("init_conv", StemBlock(in_channels, init_conv_channels))
```
* Uncomment the if stride==2 on lines 157, 168, 237, and 261 in the common.py file.
```python
        if stride == 2:
          self.maxx = nn.MaxPool2d(kernel_size=3, stride=2,padding=0)
```
```python
        if self.stride == 2:
          b = self.maxx(b)
          return x + b
```
```python
    if stride ==2:
          return ConvBlock(
             in_channels=channels,
             out_channels=channels,
             kernel_size=3,
             stride=stride,
             padding=0,
             groups=channels,
             use_bn=use_bn,
             activation=activation)
```
```python
    if stride ==2 :
        return ConvBlock(
             in_channels=channels,
             out_channels=channels,
             kernel_size=5,
             stride=stride,
             padding=1,
             groups=channels,
             use_bn=use_bn,
             activation=activation)
```
