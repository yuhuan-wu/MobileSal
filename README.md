# MobileSal

[IEEE TPAMI 2021: MobileSal: Extremely Efficient RGB-D Salient Object Detection](https://ieeexplore.ieee.org/document/9647954)

This repository contains full training & testing code, and pretrained saliency maps. We have achieved competitive performance on the RGB-D salient object detection task with a speed of 450fps.

If you run into any problems or feel any difficulties to run this code, do not hesitate to leave issues in this repository.

My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

[[PDF]](https://mftp.mmcheng.net/Papers/21PAMI_MobileSal.pdf)

### Requirements

* Python 3.7+
* Jittor
* Tested on Jittor 1.3.1, [Check Complete Envs](envs/)

Note: `Jittor` requires Python3.7+. For Jittor users, please run `git checkout jittor` first.

### Installing

For installing Jittor, please run the following commands:

````
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install -r requirements.txt
# 如果您电脑包含Nvidia显卡，检查cudnn加速库
python3.7 -m jittor.test.test_cudnn_op
python3.7 -m pip install requ
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

### Pretrained Models

As in our paper, we train our model on the NJU2K_NLPR training set, and test our model on NJU2K_test, NLPR_test, STEREO, SIP, and SSD datasets. For DUTLF-D, we train our model on DUTLF-D training set and evaluate on its testing test.

(Default) Trained on NJU2K_NLPR training set: 
* Single-scale Training: [[Google Drive]](https://drive.google.com/file/d/1dfyFkdsI1rOfmhmgG-o45ggnOj5Wpr1d/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)
* Multi-scale Training: [[Google Drive]](https://drive.google.com/file/d/1WTRxxO78wx48F3ItfXG8vbSL4IvWanyr/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

(Custom) Training on DUTLF-D training set:
* Multi-scale Training: [[Google Drive]](https://drive.google.com/file/d/1L26kN_sZkLVDBzh_NOCB-ajkrGJdIovi/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

Download them and put them into the `pretrained/` folder.

### Test / Evaluation / Results

After preparing the pretrained models, it is also very simple to test our network:

```
bash ./tools/test.sh
```

The scripts will automatically generate saliency maps on the `maps/` directory.

### Pretrained Saliency maps

For covenience, we provide the pretrained saliency maps on several datasets as below:

* Single-scale Training: [[Google Drive]](https://drive.google.com/file/d/1UA7zZmMO1Js0Jh9VQwo5JjYRF3qX0y0N/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

* Multi-scale Training: [[Google Drive]](https://drive.google.com/file/d/1-vwtUPh3UWez963IyZNO6HZkGdC3GusL/view?usp=sharing), [[Baidu Pan, 9nxi]](https://pan.baidu.com/s/1a71BlcvX0MTBuP_GGd84WA)

### TODO

1. Release the pretrained models and saliency maps on COME15K dataset.
2. Release the ONNX model for real-world applications.
3. Add results with the [P2T](https://arxiv.org/abs/2106.12011) transformer backbone.

### Other Tips

* I encourage everyone to contact me via my e-mail. My e-mail is: wuyuhuan @ mail.nankai (dot) edu.cn

### License

The code is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for NonCommercial use only.


### Citations

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


### Acknowlogdement

This repository is built under the help of the following five projects for academic use only:

* [PyTorch](https://github.com/pytorch/pytorch)

* [Jittor](https://github.com/Jittor/jittor)

