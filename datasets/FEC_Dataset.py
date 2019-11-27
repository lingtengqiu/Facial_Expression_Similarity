# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher Lingteng Qiu
import torch
import torch.utils.data
from PIL import Image
import os
from utils.mc_reader import MemcachedReader
import io

class FEC_Dataset(torch.utils.data.Dataset):
    def __init__(self,root,transforms,ann_file):
        super(FEC_Dataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.ann_file = ann_file
        assert os.path.exists(self.root),"{} no exist".format(self.root)
        assert os.path.exists(self.ann_file),"{} no exist".format(self.ann_file)
        self.triple_sets=[]
        self.labels = []
        self.types = []
        self.reader = MemcachedReader()

        with open(self.ann_file) as reader:
            for line in reader:
                keys = line.strip().split(",")
                self.triple_sets.append([*keys[:3]])
                self.labels.append(int(keys[3]))
                self.types.append(keys[4])
    def open(self,img):
        try:
            filebytes = self.reader(img)
            buff = io.BytesIO(filebytes)
            image = Image.open(buff).convert('RGB')
        except:
            img = img
            image = Image.open(img).convert('RGB')
        return image
    def __getitem__(self, item):
        ids = self.triple_sets[item]
        label = self.labels[item]
        types = self.types[item]
        img_ids = [os.path.join(self.root,id) for id in ids]
        imgs = [self.open(id) for id in img_ids]
        if self.transforms is not None:
            img_tensors = [self.transforms(img) for img in imgs]
        return img_tensors,label,types
    def __len__(self):
        return len(self.labels)

