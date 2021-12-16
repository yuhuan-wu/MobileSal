import torch
import cv2
import torch.utils.data
import torch.nn.functional as F
import random

class Dataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, dataset, file_root='data/', transform=None, use_depth=False, depth_postfix="_depth.png"):
        """
        dataset: dataset name, e.g. NJU2K_NLPR_train
        file_root: root of data_path, e.g. ./data/
        """
        self.file_list = open(file_root + '/' + dataset + '.txt').read().splitlines()
        self.images = [file_root + '/' + x.split(' ')[0] for x in self.file_list]
        self.gts = [file_root + '/' +  x.split(' ')[1] for x in self.file_list]
        self.transform = transform
        self.use_depth = use_depth
        self.depth_postfix = depth_postfix

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label_name = self.gts[idx]
        depth_name = image_name[:-4] + self.depth_postfix
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        depth = cv2.imread(depth_name, 0) if self.use_depth else None
        if self.transform:
            [image, label, depth] = self.transform(image, label, depth)
        if self.use_depth:
            return image, label, depth
        else:
            return image, label

    def get_img_info(self, idx):
        img = cv2.imread(self.images[idx])
        return {"height": img.shape[0], "width": img.shape[1]}

