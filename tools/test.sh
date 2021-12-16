#!/bin/bash
PREFIX=./pretrained/
MODEL_NAME=mobilesal_ss.pth
MODEL_PATH=$PREFIX$MODEL_NAME

python3 tools/test.py --pretrained $MODEL_PATH \
                                      --savedir ./maps/$MODEL_NAME/ \
                                      --depth 1


