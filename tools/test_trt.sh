#!/bin/bash
PREFIX=./pretrained/
MODEL_NAME=mobilesal_ss
MODEL_PATH=$PREFIX$MODEL_NAME.pth

python3 tools/test_trt.py --pretrained $MODEL_PATH \
                                      --savedir ./maps_trt/$MODEL_NAME/ \
                                      --depth 1



