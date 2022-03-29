# Swin-BOAT

This is developed based on the official version of [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
We only change ./model/swin_transformer.py to ./model/boat_swin_transformer.py and keep other codes unchanged.


## Start

Please refer to [Start for Swin](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) for installing the prerequisite.

## Training

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

## Evaluation

To evaluate a pre-trained `BOAT-Swin Transformer` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path> 
```

## Pre-trained models

[BOAT-Swin-Tiny](https://www.dropbox.com/s/xa94uewsrvjglnn/tiny.pth?dl=0)

[BOAT-Swin-Small](https://www.dropbox.com/s/7ih1zvii3bvdcgd/small.pth?dl=0)

[BOAT-Swin-Base](https://www.dropbox.com/s/70hr7h0smcr0gr9/base.pth?dl=0)

## Acknowledgement
This is developed based on the official version of [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
