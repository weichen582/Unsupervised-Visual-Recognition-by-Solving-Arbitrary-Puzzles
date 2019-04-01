#!/bin/sh

python main.py \
  --task puzzle_eval \
  \
  --dataset imagenet \
  --val_paths_file files/val_cls.txt \
  \
  --preprocessing resize_small,central_crop,crop_patches \
  --smaller_size 256 \
  --crop_size 255 \
  --patch_size 64 \
  --cell_size 85 \
  --channel_jitter 0 \
  \
  --backbone alexnet \
  "$@"
