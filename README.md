This repository provides code for paper 《Lightweight Deep Neural Network Model With Padding-free Downsampling》

Experiments with BSConv
--
__《Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets》<br>__
Daniel Haase∗ &nbsp; Manuel Amthor∗ <br>
ZEISS Microscopy  &nbsp; ZEISS Microscopy


Following the hyperparameter settings of Haase et al, the padding-free downsampling and hybrid stem modules proposed in this article were experimented on CIFAR-100, Stanford Dogs, and ImageNet datasets.

## Orig
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 3.066M | 68.5M |  75.37% |  
| MobileNetv3-large(BSConv-S) | 3.066M | 68.5M |  77.87% |  
| ResNet-20 | 0.278M | 41.4M |  68.12% |  
| ResNet-110(BSConv-U) | 0.245M | 41.8M |  71.58% |  
| WideResNet-40-3 | 5.056M | 735.8M |  76.23% |  
| WideResNet-40-8(BSConv-U) | 4.286M | 675.1M |  77.79% |  

## Ours
| Model | Parameters | FLOPs |  Accuracy |
| :---: | :---: | :---: | :---: |   
| MobileNetv3-large | 3.067M | 54.6M ↓|  75.71% |  
| MobileNetv3-large(BSConv-S) | 3.067M | 54.6M ↓|  78.36% |  
| ResNet-20 | 0.282M | 37.8M ↓|  68.30% |  
| ResNet-110(BSConv-U) | 0.249M | 38.6M ↓|  71.62% |  
| WideResNet-40-3 | 5.287M | 668.7M ↓|  76.28% |  
| WideResNet-40-8(BSConv-U) | 4.457M | 615.6M ↓|  78.05% |  
