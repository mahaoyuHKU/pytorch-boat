# BOAT: Bilateral Local Attention Vision Transformer


This is develppped based on [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
We only change ./model/swin_transformer.py to ./model/boat_swin_transformer.py and keep other codes unchanged.


# Start

Please refer to [Start for Swin](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) for installing the prerequisite.

# Training

`BOAT-Swin-T`:

```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--cfg configs/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 256
```

`BOAT-Swin-S`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin_small_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 
```

`BOAT-Swin-B`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/swin_base_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128 \
```

# Evaluation

To evaluate a pre-trained `BOAT-Swin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

# Pre-trained models

[BOAT-Swin-Tiny](https://www.dropbox.com/s/xa94uewsrvjglnn/tiny.pth?dl=0)

[BOAT-Swin-Small](https://www.dropbox.com/s/7ih1zvii3bvdcgd/small.pth?dl=0)

[BOAT-Swin-Base](https://www.dropbox.com/s/70hr7h0smcr0gr9/base.pth?dl=0)

[BOAT-CSwin-Tiny](https://www.dropbox.com/s/rsmtu6r0v2lt0y5/cswin_tiny.pth.tar?dl=0)

[BOAT-CSwin-Small](https://www.dropbox.com/s/cnl00d1faxxoi19/cswin_small.pth.tar?dl=0)

[BOAT-CSwin-Base](https://www.dropbox.com/s/92sr8r8zhng1mqg/cswin_base.pth.tar?dl=0)
# CSWin-Transformer

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cswin-transformer-a-general-vision/semantic-segmentation-on-ade20k)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k?p=cswin-transformer-a-general-vision)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cswin-transformer-a-general-vision/semantic-segmentation-on-ade20k-val)](https://paperswithcode.com/sota/semantic-segmentation-on-ade20k-val?p=cswin-transformer-a-general-vision)

This repo is the official implementation of ["CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows"](https://arxiv.org/pdf/2107.00652.pdf). 

## Introduction


## Requirements

timm==0.3.4, pytorch>=1.4, opencv, ... , run:

```
bash install_req.sh
```

Apex for mixed precision training is used for finetuning. To install apex, run:

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train

Train the three lite variants: CSWin-Tiny, CSWin-Small and CSWin-Base:
```
bash train.sh 8 --data <data path> --model CSWin_64_12211_tiny_224 -b 256 --lr 2e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.2
```
```
bash train.sh 8 --data <data path> --model CSWin_64_24322_small_224 -b 256 --lr 2e-3 --weight-decay .05 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99984 --drop-path 0.4
```
```
bash train.sh 8 --data <data path> --model CSWin_96_24322_base_224 -b 128 --lr 1e-3 --weight-decay .1 --amp --img-size 224 --warmup-epochs 20 --model-ema-decay 0.99992 --drop-path 0.5
```





## Acknowledgement
This is developped based on CSWin Transformer


# If you use this code for your research, please consider citing:

```bash
@article{BOAT,
  author    = {Tan Yu and Gangming Zhao and Ping Li and Yizhou Yu},
  title     = {{BOAT:} Bilateral Local Attention Vision Transformer},
  journal   = {CoRR},
  volume    = {abs/2201.13027},
  year      = {2022},
  url       = {https://arxiv.org/abs/2201.13027},
}
```
