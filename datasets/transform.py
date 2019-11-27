# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher LingtengQiu
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size,mean_pixel):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.mean_pixel = list(map(int,mean_pixel))

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
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

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        h_pad = self.max_size - size[0]
        w_pad = self.max_size - size[1]
        if h_pad<0 or w_pad<0:
            image = F.resize(image,(self.max_size,self.max_size))
        else:
            image = np.asarray(image)
            r  = np.pad(image[:,:,0:1],((0,h_pad),(0,w_pad),(0,0)),'constant',constant_values=(self.mean_pixel[2],self.mean_pixel[2]))
            g  = np.pad(image[:,:,1:2],((0,h_pad),(0,w_pad),(0,0)),'constant',constant_values=(self.mean_pixel[1],self.mean_pixel[1]))
            b  = np.pad(image[:,:,2:3],((0,h_pad),(0,w_pad),(0,0)),'constant',constant_values=(self.mean_pixel[0],self.mean_pixel[0]))
            image = np.concatenate([r,g,b],axis=2)
            image = Image.fromarray(image)
        return image


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            image = F.hflip(image)
        return image


class ColorJitter(object):
    def __init__(self,
                 brightness=None,
                 contrast=None,
                 saturation=None,
                 hue=None,
                 ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,)

    def __call__(self, image):
        image = self.color_jitter(image)
        return image


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image
def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )
    color_jitter = ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = Compose(
        [
            color_jitter,
            Resize(min_size, max_size,cfg.INPUT.PIXEL_MEAN),
            RandomHorizontalFlip(flip_prob),
            ToTensor(),
            normalize_transform,
        ]
    )
    return transform
