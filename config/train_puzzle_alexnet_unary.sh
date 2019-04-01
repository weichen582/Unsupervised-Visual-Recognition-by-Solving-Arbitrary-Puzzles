#!/bin/sh

python main.py \
  --task puzzle_train \
  \
  --dataset imagenet \
  --train_paths_file files/train_cls.txt
  --val_paths_file files/val_sub_cls.txt
  \
  --preprocessing resize_small,central_crop,crop_patches \
  --smaller_size 256 \
  --crop_size 255 \
  --patch_size 64 \
  --cell_size 85 \
  --channel_jitter 5 \
  \
  --backbone alexnet \
  \
  --base_lr 0.01 \
  --epochs 50 \
  "$@"