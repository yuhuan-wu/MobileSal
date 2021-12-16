#!/bin/bash
PREFIX=./pretrained/
MODEL_NAME=mobilesal_dut_ms
MODEL_PATH=$PREFIX$MODEL_NAME.pth


python3 tools/test.py --pretrained $MODEL_PATH \
                      --dutlf_test 1 \
                                      --savedir ./maps/$MODEL_NAME/ \
                                      --depth 1

                                   


