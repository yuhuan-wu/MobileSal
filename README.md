# MobileSal

[IEEE TPAMI 2021: MobileSal: Extremely Efficient RGB-D Salient Object Detection](https://ieeexplore.ieee.org/document/9647954)

This repository contains full training & testing code, and pretrained saliency maps. We have achieved competitive performance on the RGB-D salient object detection task with a speed of 450fps.

If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository.

My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

[[PDF]](https://mftp.mmcheng.net/Papers/21PAMI_MobileSal.pdf)

This repository contains:

- [x] Full code, data, pretrained models for `training` and `testing`
- [x] MobileSal deployment, achieving 420FPS (fp32) and 800FPS (fp16) with `batch size 1` on a single RTX 2080Ti.

* [Requirements](#requirements)
  * [PyTorch](#pytorch)
  * [Jittor](#jittor)
  * [Deployment](#deployment)
* [`Data Preparing`](#data-preparing)
* [`Train`](#train)
* [`Test`](#test)
  * [Pretrained Models](#pretrained-models)
  * [Generate Saliency Maps](#generate-saliency-maps)
  * [Deployment](#deployment-1)
  * [Speed Test](#speed-test)
  * [`Pretrained Saliency maps`](#pretrained-saliency-maps)
* [Others](#others)
  * [TODO](#todo)
  * [Contact](#contact)
  * [License](#license)
  * [Citation](#citation)
  * [Acknowlogdement](#acknowlogdement)

### Requirements

#### PyTorch 

* Python 3.6+
* PyTorch >=0.4.1, OpenCV-Python
* Tested on PyTorch 1.7.1

#### Jittor

* Python 3.7+
* Jittor, OpenCV-Python
* Tested on Jittor 1.3.1

#### Deployment

* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)

For Jittor users, we create a branch `jittor`. So please run the following command first:

````
git checkout jittor
````

To install MobileSal, please run:

````
pip install -r envs/requirements.txt
````

### Data Preparing

Before training/testing our network, please download the training data: 

* Preprocessed data of 6 datasets: [[Google Drive]](https://drive.google.com/file/d/1czlZyW9_6k3ueS--TDAZK6M7Uv6FpUfO/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)


Note: if you are blocked by Google and Baidu services, you can contact me via e-mail and I will send you a copy of data and model weights.

We have processed the data to json format so you can use them without any preprocessing steps. 
After completion of downloading, extract the data and put them to `./data/` folder.
Then, the `./datasets/` folder should contain six folders: `NJU2K/, NLPR/, STERE/, SSD/, SIP/, DUT-RGBD/`, representing `NJU2K, NLPR, STEREO, SSD, SIP, DUTLF-D` datasets, respectively.


### Train

It is very simple to train our network. We have prepared a script to run the training step:
```
bash ./tools/train.sh
```

### Test


#### Pretrained Models

As in our paper, we train our model on the NJU2K_NLPR training set, and test our model on NJU2K_test, NLPR_test, STEREO, SIP, and SSD datasets. For DUTLF-D, we train our model on DUTLF-D training set and evaluate on its testing test.

(Default) Trained on NJU2K_NLPR training set: 
* Single-scale Training: [[Google Drive]](https://drive.google.com/file/d/1dfyFkdsI1rOfmhmgG-o45ggnOj5Wpr1d/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)
* Multi-scale Training: [[Google Drive]](https://drive.google.com/file/d/1WTRxxO78wx48F3ItfXG8vbSL4IvWanyr/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

(Custom) Training on DUTLF-D training set:
* Multi-scale Training: [[Google Drive]](https://drive.google.com/file/d/1L26kN_sZkLVDBzh_NOCB-ajkrGJdIovi/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

Download them and put them into the `pretrained/` folder.


#### Generate Saliency Maps

After preparing the pretrained models, it is also very simple to generate saliency maps via MobileSal:

```
bash ./tools/test.sh
```

The scripts will automatically generate saliency maps on the `maps/` directory.

#### Deployment

The deployment largely speeds up MobileSal with batch size of 1.
An example script is located at: `tools/test_trt.sh`. Run:

```
bash ./tools/test_trt.sh
```

This script will automatically convert PyTorch MobileSal to TensorRT-based MobileSal. Then it will generate saliency maps via the TensorRT-based MobileSal.
On deployment for real-world applications, you can load the converted TensorRT MobileSal for inference:

```
from torch2trt import torch2trt, TRTModule
model = TRTModule(); trt_model_path = "pretrained/mobilesal_trt.pth"
model.load_state_dict(torch.load(trt_model_path))
result = model(image, depth) # get result with [torch.Tensor] input
```

#### Speed Test
We provide a speed test script on MobileSal:

```
python speed_test.py
```

The speed result on a single RTX 2080Ti is as below:

|     Type     | Input  Size    | Batch Size | FP16 | FPS | 
|-----------------|:---------:|:-----:|:-----:|:-----------:|
| PyTorch   | 320 x 320 |  20 |  No |     450    |     
| TensorRT | 320 x 320 |  1 |  No |     420    |    
| TensorRT | 320 x 320 |  1 |  Yes |     800    |     


#### Pretrained Saliency maps

For covenience, we provide the pretrained saliency maps on several datasets as below:

* Single-scale Training: [[Google Drive]](https://drive.google.com/file/d/1UA7zZmMO1Js0Jh9VQwo5JjYRF3qX0y0N/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

* Multi-scale Training: [[Google Drive]](https://drive.google.com/file/d/1-vwtUPh3UWez963IyZNO6HZkGdC3GusL/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

### Others 

#### TODO

1. Release the pretrained models and saliency maps on COME15K dataset.
2. Add results with the [P2T](https://arxiv.org/abs/2106.12011) transformer backbone.

#### Contact

* I encourage everyone to contact me via my e-mail. My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

#### License

The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for NonCommercial use only.


#### Citation

If you are using the code/model/data provided here in a publication, please consider citing our work:

````
@ARTICLE{wu2021mobilesal,
  author={Wu, Yu-Huan and Liu, Yun and Xu, Jun and Bian, Jia-Wang and Gu, Yu-Chao and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={MobileSal: Extremely Efficient RGB-D Salient Object Detection}, 
  year={2021},
  doi={10.1109/TPAMI.2021.3134684}
}
````


#### Acknowlogdement

This repository is built under the help of the following five projects for academic use only:

* [PyTorch](https://github.com/pytorch/pytorch)

* [Jittor](https://github.com/Jittor/jittor)

* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
