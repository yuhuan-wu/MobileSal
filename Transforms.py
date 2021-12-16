import numpy as np
import torch
import random
import cv2
import torch.nn.functional as F

class Scale(object):
    """
    Resize the given image to a fixed scale
    """
    def __init__(self, wi, he):
        '''
        :param wi: width after resizing
        :param he: height after reszing
        '''
        self.w = wi
        self.h = he

    # modified from torchvision to add support for max size


    def __call__(self, img, label, depth=None):
        '''
        :param img: RGB image
        :param label: semantic label image
        :return: resized images
        '''
        # bilinear interpolation for RGB image
        img = cv2.resize(img, (self.w, self.h))
        # nearest neighbour interpolation for label image
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (self.w, self.h)) if depth is not None else None
        return [img, label, depth]


class Resize(object):
    def __init__(self, min_size, max_size, strict=False):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.strict = strict

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if not self.strict:
            size = random.choice(self.min_size)
            max_size = self.max_size
            if max_size is not None:
                min_original_size = float(min((w, h)))
                max_original_size = float(max((w, h)))
                if max_original_size / min_original_size * size > max_size:
                    size = int(round(max_size * min_original_size / max_original_size))

            if (w <= h and w == size) or (h <= w and h == size):
                return (h, w)

            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)

            return (oh, ow)
        else:
            if w < h:
                return (self.max_size, self.min_size[0])
            else:
                return (self.min_size[0], self.max_size)

    def __call__(self, image, label, depth=None):
        size = self.get_size(image.shape[:2])
        #print("origin", image.shape)
        image = cv2.resize(image, size)
        #print("resized", image.shape)
        #print('*'*20)
        #I confirm that the output size is right, not reversed
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return (image, label)


class RandomCropResize(object):
    """
    Randomly crop and resize the given image with a probability of 0.5
    """
    def __init__(self, crop_area):
        '''
        :param crop_area: area to be cropped (this is the max value and we select between 0 and crop area
        '''
        self.cw = crop_area
        self.ch = crop_area

    def __call__(self, img, label, depth=None):
        if random.random() < 0.5:
            h, w = img.shape[:2]
            x1 = random.randint(0, self.ch)
            y1 = random.randint(0, self.cw)

            img_crop = img[y1:h-y1, x1:w-x1]
            label_crop = label[y1:h-y1, x1:w-x1]

            img_crop = cv2.resize(img_crop, (w, h))
            label_crop = cv2.resize(label_crop, (w, h), interpolation=cv2.INTER_NEAREST)

            if depth is not None:
                depth_crop = depth[y1:h-y1, x1:w-x1]
                depth_crop = cv2.resize(depth_crop, (w, h))
            else:
                depth_crop = None
            return img_crop, label_crop, depth_crop
        else:
            return [img, label, depth]

class RandomFlip(object):
    """
    Randomly flip the given Image with a probability of 0.5
    """
    def __call__(self, image, label, depth=None):
        if random.random() < 0.5:
            x1 = 0 #random.randint(0, 1) # if you want to do vertical flip, uncomment this line
            if x1 == 0:
                image = cv2.flip(image, 0) # horizontal flip
                label = cv2.flip(label, 0) # horizontal flip
                depth = cv2.flip(depth, 0) if depth is not None else None
            else:
                image = cv2.flip(image, 1) # veritcal flip
                label = cv2.flip(label, 1) # veritcal flip
                depth = cv2.flip(depth, 1) if depth is not None else None
        return [image, label, depth]

class Normalize(object):
    """
    Given mean: (B, G, R) and std: (B, G, R),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.mean = mean
        self.std = std
        self.depth_mean = [0.5]
        self.depth_std = [0.5]


    def __call__(self, image, label, depth=None):
        image = image.astype(np.float32)
        image = image / 255
        label = label / 255
        for i in range(3):
            image[:,:,i] -= self.mean[i]
        for i in range(3):
            image[:,:,i] /= self.std[i]

        if depth is not None:
            depth = depth / 255
            depth -= self.depth_mean
            depth /= self.depth_std
        return [image, label, depth]

class GaussianNoise(object):
    def __init__(self, std=0.05):
        '''
        :param mean: global mean computed from dataset
        :param std: global std computed from dataset
        '''
        self.std = std

    def __call__(self, image, label, depth=None):
        noise = np.random.normal(loc=0, scale=self.std, size=image.shape)
        image = image + noise.astype(np.float32)
        return [image, label, depth]

class ToTensor(object):
    '''
    This class converts the data to tensor so that it can be processed by PyTorch
    '''
    def __init__(self, scale=1):
        '''
        :param scale: set this parameter according to the output scale
        '''
        self.scale = scale

    def __call__(self, image, label, depth=None):
        if self.scale != 1:
            h, w = label.shape[:2]
            image = cv2.resize(image, (int(w), int(h)))
            label = cv2.resize(label, (int(w/self.scale), int(h/self.scale)), \
                interpolation=cv2.INTER_NEAREST)
            depth = cv2.resize(image, (int(w), int(h))) if depth is not None else None
        image = image[:,:, ::-1].copy() # .copy() is to solve "torch does not support negative index"
        image = image.transpose((2, 0, 1))
        image_tensor = torch.from_numpy(image)
        # TODO: here, we add unsqueeze to satisfy the condition that
        # adjust_size in DataSet.py should input 4D tensor
        label_tensor =  torch.LongTensor(np.array(label, dtype=np.int)).unsqueeze(dim=0)
        if depth is not None:
            depth_tensor = torch.from_numpy(depth).unsqueeze(dim=0).float()
        else:
            depth_tensor = torch.rand(1,10,10).float()

        return [image_tensor, label_tensor, depth_tensor]


class Compose(object):
    """
    Composes several transforms together.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
