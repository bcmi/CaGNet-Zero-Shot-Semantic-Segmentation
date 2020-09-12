# CaGNet: Context-aware Feature Generation for Zero-shot Semantic Segmentation

Code for our **ACM MM 2020** paper *"Context-aware Feature Generation for Zero-shot Semantic Segmentation"*.

Created by [Zhangxuan Gu](https://github.com/zhangxgu), [Siyuan Zhou](https://github.com/Siyuan-Zhou), [Li Niu\*](https://github.com/ustcnewly), Zihan Zhao, Liqing Zhang\*.

Paper Link: [[arXiv]](http://arxiv.org/abs/2008.06893)

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{Gu2020CaGNet,
  title={Context-aware Feature Generation for Zero-shot Semantic Segmentation},
  author={Zhangxuan Gu and Siyuan Zhou and Li Niu and Zihan Zhao and Liqing Zhang},
  booktitle={ACM International Conference on Multimedia},
  year={2020}
}
```

## Introduction

Existing semantic segmentation models heavily rely on dense pixel-wise annotations. To reduce the annotation pressure, we focus on a challenging task named zero-shot semantic segmentation, which aims to segment unseen objects with zero annotations. This can be achieved by transferring knowledge across categories via semantic word embeddings. In this paper, we propose a novel context-aware feature generation method for zero-shot segmentation named as *CaGNet*. In particular, with the observation that a pixel-wise feature highly depends on its contextual information, we insert a contextual module in a segmentation network to capture the pixel-wise contextual information, which guides the process of generating more diverse and context-aware features from semantic word embeddings. Our method achieves state-of-the-art results on three benchmark datasets for zero-shot segmentation.

[![Overview of Our CaGNet](https://github.com/bcmi/CaGNet-Zero-Shot-Semantic-Segmentation/blob/master/figures/overview.JPG?raw=true)](https://github.com/bcmi/CaGNet-Zero-Shot-Semantic-Segmentation/blob/master/figures/overview.JPG?raw=true)

## Results on Pascal-Context, COCO-Stuff and Pascal-VOC

We compare our *CaGNet* with SPNet \[[github](https://github.com/subhc/SPNet), [paper](https://ieeexplore.ieee.org/document/8953827)\] and ZS3Net \[[github](https://github.com/valeoai/ZS3), [paper](https://arxiv.org/pdf/1906.00817.pdf)\].

“ST” in the following tables stands for self-training mentioned in ZS3Net. 

**Our Results on Pascal-Context dataset**

|    Method     |    hIoU    |    mIoU    | pixel acc. | mean acc.  |   S-mIoU   |   U-mIoU   |
| :-----------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|     SPNet     |     0      |   0.2938   |   0.5793   |   0.4486   |   0.3357   |     0      |
|    SPNet-c    |   0.0718   |   0.3079   |   0.5790   |   0.4488   |   0.3514   |   0.0400   |
|    ZS3Net     |   0.1246   |   0.3010   |   0.5710   |   0.4442   |   0.3304   |   0.0768   |
|  **CaGNet**   | **0.2061** | **0.3347** | **0.5924** | **0.4900** | **0.3610** | **0.1442** |
|   ZS3Net+ST   |   0.1488   |   0.3102   |   0.5725   |   0.4532   |   0.3398   |   0.0953   |
| **CaGNet+ST** | **0.2252** | **0.3352** | **0.5961** | **0.4962** | **0.3644** | **0.1630** |

**Our Results on COCO-Stuff dataset**

|    Method     |    hIoU    |    mIoU    | pixel acc. | mean acc.  |   S-mIoU   |   U-mIoU   |
| :-----------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|     SPNet     |   0.0140   |   0.3164   |   0.5132   |   0.4593   |   0.3461   |   0.0070   |
|    SPNet-c    |   0.1398   |   0.3278   |   0.5341   |   0.4363   |   0.3518   |   0.0873   |
|    ZS3Net     |   0.1495   |   0.3328   |   0.5467   |   0.4837   |   0.3466   |   0.0953   |
|  **CaGNet**   | **0.1819** | **0.3345** | **0.5658** | **0.4845** | **0.3549** | **0.1223** |
|   ZS3Net+ST   |   0.1620   |   0.3367   |   0.5631   | **0.4862** |   0.3489   |   0.1055   |
| **CaGNet+ST** | **0.1946** | **0.3372** | **0.5676** |   0.4854   | **0.3555** | **0.1340** |

**Our Results on Pascal-VOC dataset**

|    Method     |    hIoU    |    mIoU    | pixel acc. | mean acc.  |   S-mIoU   |   U-mIoU   |
| :-----------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|     SPNet     |   0.0002   |   0.5687   |   0.7685   |   0.7093   |   0.7583   |   0.0001   |
|    SPNet-c    |   0.2610   |   0.6315   |   0.7755   |   0.7188   |   0.7800   |   0.1563   |
|    ZS3Net     |   0.2874   |   0.6164   |   0.7941   |   0.7349   |   0.7730   |   0.1765   |
|  **CaGNet**   | **0.3972** | **0.6545** | **0.8068** | **0.7636** | **0.7840** | **0.2659** |
|   ZS3Net+ST   |   0.3328   |   0.6302   |   0.8095   |   0.7382   |   0.7802   |   0.2115   |
| **CaGNet+ST** | **0.4366** | **0.6577** | **0.8164** | **0.7560** | **0.7859** | **0.3031** |

Please note that our reproduced results of SPNet on Pascal-VOC dataset are obtained using their released model and code with careful tuning, but still lower than their reported results.

## Hardware Dependency

Our released code temporarily supports a single GPU or multiple GPUs. To acquire satisfactory training results, we advise that each GPU card should be at least 32GB with batch size larger than 8.

The results in the conference paper / this repository are obtained on a single 32GB GPU with batch size 8. If you use multiple GPUs (each ≥ 32GB) to train *CaGNet*, you might hopefully achieve better results.

## Getting Started

### Installation

1.Clone this repository.

```
git clone https://github.com/bcmi/CaGNet-Zero-Shot-Semantic-Segmentation.git
```

2.Create python environment for *CaGNet* via conda.

```
conda env create -f CaGNet_environment.yaml
```

3.Download dataset.

  - Pascal-VOC

    --> **CaGNet_VOC2012_data.tar** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/HBqSsFWTl) or [BaiduNetDisk](https://pan.baidu.com/s/17aEkQuwL7VQRSACUV97Pkw) (extraction code: *beau*)
    1) download the above .tar file into directory **./dataset/voc12/**
    2) uncompress it to form **./dataset/voc12/images/** and **./dataset/voc12/annotations/**
  - Pascal-Context

    --> **CaGNet_context_data.tar** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/KOE8KxwrU) or [BaiduNetDisk](https://pan.baidu.com/s/11f22mnXQRGAR78QR8-W9ow) (extraction code: *rk29*)
    1) download the above .tar file into directory **./dataset/context/**
    2) uncompress it to form **./dataset/context/images/** and **./dataset/context/annotations/**
  - COCO-Stuff
    1) follow the setup instructions on the [COCO-Stuff homepage](https://github.com/nightrome/cocostuff) to obtain two folders: **images** and **annotations**.
    2) move the above two folders into directory **./dataset/cocostuff/** to form **./dataset/cocostuff/images/** and **./dataset/cocostuff/annotations/**

4.Download pre-trained weights and our optimal models into directory **./trained_models/**

  - deeplabv2 pretrained weight for Pascal-VOC and Pascal-Context

    --> **deeplabv2_resnet101_init.pth** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/y8H4qedH2) or [BaiduNetDisk](https://pan.baidu.com/s/1N0spp4zKBWpo6pD2kCbqeA) (extraction code: *5o0m*) 
  
  - SPNet pretrained weight for COCO-Stuff

    --> **spnet_cocostuff_init.pth** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/HjIHArLR4) or [BaiduNetDisk](https://pan.baidu.com/s/18OQF-J-Hlj-uZHf4qcZ5TA) (extraction code: *qjpo*)
  
  - our best model on Pascal-VOC

    --> **voc12_ourbest.pth** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/RLYautiFq) or [BaiduNetDisk](https://pan.baidu.com/s/11npWXmwMNLpOfj0wjWOiGg) (extraction code: *nxj4*)
    
  - our best model on Pascal-Context

    --> **context_ourbest.pth** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/efHuY6bsz) or [BaiduNetDisk](https://pan.baidu.com/s/1-ULedCAlo16kmbJKUrjsAQ) (extraction code: *0x2i*)

  - our best model on COCO-Stuff

    --> **cocostuff_ourbest.pth** : [BCMI-Cloud](https://bcmi.cloud:5001/sharing/yq3H8MX06) or [BaiduNetDisk](https://pan.baidu.com/s/1lLm1ovZszEOo6Myu-RoU4w) (extraction code: *xl88*)

### Training

1.Train on Pascal-VOC dataset

```
python train.py --config ./configs/voc12.yaml --schedule step1
python train.py --config ./configs/voc12_finetune.yaml --schedule mixed
```

2.Train on Pascal-Context dataset

```
python train.py --config ./configs/context.yaml --schedule step1
python train.py --config ./configs/context_finetune.yaml --schedule mixed
```

3.Train on COCO-Stuff dataset

```
python train.py --config ./configs/cocostuff.yaml --schedule step1
python train.py --config ./configs/cocostuff_finetune.yaml --schedule mixed
```

### Testing

1.Test our best model on Pascal-VOC dataset

```
python train.py --config ./configs/voc12.yaml --init_model ./trained_models/voc12_ourbest.pth --val
```

2.Test our best model on Pascal-Context dataset

```
python train.py --config ./configs/context.yaml --init_model ./trained_models/context_ourbest.pth --val
```

3.Test our best model on COCO-Stuff dataset

```
python train.py --config ./configs/cocostuff.yaml --init_model ./trained_models/cocostuff_ourbest.pth --val
```

## Acknowledgement

Some of the codes are built upon [FUNIT](https://github.com/NVlabs/FUNIT) and [SPNet](https://github.com/subhc/SPNet). Thanks them for their great work!

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request!

*CaGNet* is freely available for non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.