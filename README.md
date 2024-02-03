This repository provides code for paper 《Lightweight Deep Neural Network Model With Padding-free Downsampling》

__《Rethinking Depthwise Separable Convolutions: How Intra-Kernel Correlations Lead to Improved MobileNets》<br>__
Daniel Haase∗ &nbsp; Manuel Amthor∗ <br>
ZEISS Microscopy  &nbsp; ZEISS Microscopy

Following the hyperparameter settings of Haase et al, the padding-free downsampling and hybrid stem modules proposed in this article were experimented on CIFAR-100, Stanford Dogs, and ImageNet datasets.

For the CIFAR-100 dataset, using the first set of hyperparameters
--

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

Ablation Experiments
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
