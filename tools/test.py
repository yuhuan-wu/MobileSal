import sys
sys.path.insert(0, '.')

import torch
import jittor as jt
jt.flags.use_cuda = 1

import jittor.nn as nn
import cv2
import time
import os
import os.path as osp
import numpy as np
from argparse import ArgumentParser
from SalEval import SalEval
from models import model as net
import matplotlib.pyplot as plt
from tqdm import tqdm




def get_mean_set(args):
    # for DUTS training dataset
    mean = [0.406, 0.456, 0.485] #BGR
    std = [0.225, 0.224, 0.229]
    return mean, std



def psnr(dt, gt):
    img1 = dt * 255; img2 = gt * 255
    mse = torch.mean((img1 - img2) ** 2)
    return (20 * torch.log10(255.0 / torch.sqrt(mse))).item()



@torch.no_grad()
def validateModel(args, model, image_list, label_list, savedir):
    mean, std = get_mean_set(args)
    evaluate = SalEval()

    for idx in tqdm(range(len(image_list))):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255
        depth = cv2.imread(image_list[idx][:-4] + "_depth.png", 0) / 255
        if args.depth:
            depth -= 0.5
            depth /= 0.5
            depth = cv2.resize(depth, (args.inWidth, args.inHeight))
            depth = jt.float32(depth).view(1, 1, args.inWidth, args.inHeight)
        else:
            depth = None
        
        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(image, (args.inWidth, args.inHeight))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std

        img = img[:,:, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img_tensor = jt.float32(img).unsqueeze(dim=0)
        #img_tensor = torch.from_numpy(img)
        #img_tensor = torch.unsqueeze(img_tensor, 0)# add a batch dimension

        label = jt.float32(label).unsqueeze(dim=0)
        plt.axis('off')
        fig = plt.gcf()

        img_out = model(img_tensor, depth=depth)


        img_out = nn.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        if args.save_depth:
            depth_out = nn.interpolate(depth_out, size=image.shape[:2], mode='bilinear', align_corners=False)
            depthMap_numpy = (depth_out * 255).data.cpu().numpy()[0, 0].astype(np.uint8)
            depthMapGT_numpy = ((depth[0,0] *0.5 + 0.5) * 255).cpu().numpy().astype(np.uint8)
        
        if args.vis_feats:
            fig.savefig("{}".format("maps/visualize/"+os.path.basename(image_list[idx][:-4]+'.jpg')), format='jpg', transparent=True, dpi=300, pad_inches=0)
            plt.close('all')
     
        evaluate.addBatch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

        salMap_numpy = (img_out*255).numpy()[0,0].astype(np.uint8)

        name = image_list[idx].split('/')[-1]
        if False:
            cv2.imwrite(osp.join(savedir, name[:-4] + '.png'), salMap_numpy)
        if args.save_depth:
            cv2.imwrite(osp.join(savedir, name[:-4] + '_depth_pred.png'), depthMap_numpy)
            cv2.imwrite(osp.join(savedir, name[:-4] + '_depth.png'), depthMapGT_numpy)

    F_beta, MAE = evaluate.getMetric()
    print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))

def main(args, data_list):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, data_list + '.txt')) as textFile:
        for line in textFile:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + line_arr[1].strip())

    model = net.MobileSal()
    #model = torch.nn.DataParallel(model)
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(args.pretrained, map_location=torch.device('cpu'))
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        print("*"*15+"\n\n"+"*"*15)
        print("loaded args:", args, '\n', "not exactly match params")
        model.load_state_dict(state_dict, strict=False)
        print("*"*15+"\n\n"+"*"*15)

    # set to evaluation mode
    model.eval()
    
    savedir = args.savedir + '/' + data_list + '/'
    if not osp.isdir(savedir):
        os.makedirs(savedir)

    validateModel(args, model, image_list, label_list, savedir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=320, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=320, help='Height of RGB image')
    parser.add_argument('--savedir', default='./maps/', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default="../pretrained/mobilesal_ss.pth", help='Pretrained model')
    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--save_depth', default=0, type=int)

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    data_lists = ['NJU2K_test', 'NLPR_test', 'SSD', 'STERE', 'SIP', 'DES', 'LFSD']
    for data_list in data_lists:
        print("processing ", data_list)
        main(args, data_list)
