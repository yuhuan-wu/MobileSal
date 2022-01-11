#!/bin/bash
SAVE_PREFIX=./snapshots/ss_train/

SAVE_PATH=$SAVE_PREFIX

CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --file_root ./data/ \
                                         --max_epochs 60 \
                                         --num_workers 2 \
                                         --batch_size 10 \
                                         --savedir $SAVE_PATH \
                                         --depth 1 \
                                         --lr_mode poly \
                                         --lr 1e-4 \
                                         --inWidth 320 \
                                         --inHeight 320 \
                                         --ms 0
                                         

