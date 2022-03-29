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