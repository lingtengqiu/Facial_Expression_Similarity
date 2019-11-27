# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher LingtengQiu
import sys
sys.path.append("./")
import argparse
from configs import cfg
from models import build_model
import torch
import torch.nn as nn
import imp
from datasets.build import build_dataset
from datasets.transform import  build_transforms
import os
import fire
import cv2
import numpy as np
import tqdm
from utils import loss_opts
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from utils.board import Visualizer
from utils.distributed import AverageMeter
import time
resume_dir = "../Face_Emotion_Net_resnet_sigm/"
def make_dataset(cfg,is_train = True):
    paths_catalog = import_file(
        "configs.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog

    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    transforms = build_transforms(cfg,is_train)
    train_data_sets = build_dataset(dataset_list,transforms,DatasetCatalog,True)
    return train_data_sets
def make_dataloaders(cfg,datasets,is_train = True):
    dataloaders =[]
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset,batch_size=cfg.SOLVER.IMS_PER_BATCH,shuffle=is_train))
    return dataloaders
def mean_std(src):
    img_list = os.listdir(src)
    b = 0.
    g = 0.
    r = 0.
    for name in tqdm.tqdm(img_list):
        name = os.path.join(src,name)
        img = cv2.imread(name)
        b += np.mean(img[:,:,0])
        g += np.mean(img[:,:,1])
        r += np.mean(img[:,:,2])
    b = b/len(img_list)
    g = g/len(img_list)
    r = r/len(img_list)
    print(b,g,r)



def import_file(module_name, file_path, make_importable=None):
    module = imp.load_source(module_name, file_path)
    return module
def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    viewer = Visualizer(cfg.OUTPUT_DIR)
    #Model
    model = build_model(cfg)
    model = DataParallel(model).cuda()
    if cfg.MODEL.WEIGHT !="":
        model.module.backbone.load_state_dict(torch.load(cfg.MODEL.WEIGHT))
        #freeze backbone
        for key,val in model.module.backbone.named_parameters():
            val.requires_grad = False


    batch_time = AverageMeter()
    data_time = AverageMeter()

    #optimizer
    optimizer = getattr(torch.optim,cfg.SOLVER.OPTIM)(model.parameters(),lr = cfg.SOLVER.BASE_LR,weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg.SOLVER.STEPS,gamma= cfg.SOLVER.GAMMA)

    #dataset
    datasets  = make_dataset(cfg,is_train=False)
    dataloaders = make_dataloaders(cfg,datasets,False)
    iter_epoch = (cfg.SOLVER.MAX_ITER)//len(dataloaders[0])+1
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    ite = 0
    batch_it = [i *cfg.SOLVER.IMS_PER_BATCH for i in range(1,4)]


    # start time
    start = time.time()
    inference_list = ['resnet18_14.pth','resnet18_13.pth','resnet18_12.pth','resnet18_11.pth','resnet18_10.pth']
    for inference_weight in inference_list:
        model.load_state_dict(torch.load(os.path.join(resume_dir,inference_weight)))
        model.eval()

        total_count = 0
        one_count = 0
        two_count = 0
        three_count = 0
        one_number = 0
        two_number = 0
        three_number = 0
        for dataloader in dataloaders:
            for imgs,labels,types in tqdm.tqdm(dataloader,desc="dataloader:"):
                types = np.asarray(types)
                lr_sche.step()
                data_time.update(time.time() - start)

                inputs = torch.cat([imgs[0].cuda(),imgs[1].cuda(),imgs[2].cuda()],dim=0)
                with torch.no_grad():
                    features = model(inputs)
                acc,batch_loss = loss_opts.batch_triple_loss_acc(features,labels,types,size_average=True)
                print(batch_loss)
                xxx
                total_count+= batch_loss.shape[0]-acc

                ONE_CLASS = (batch_loss[np.nonzero(types=='ONE_CLASS_TRIPLET')[0]])
                TWO_CLASS = (batch_loss[np.nonzero(types=='TWO_CLASS_TRIPLET')[0]])
                THREE_CLASS = (batch_loss[np.nonzero(types=='THREE_CLASS_TRIPLET')[0]])
                one_count += ONE_CLASS.shape[0] - torch.nonzero(ONE_CLASS).shape[0]
                two_count += TWO_CLASS.shape[0] - torch.nonzero(TWO_CLASS).shape[0]
                three_count += THREE_CLASS.shape[0] - torch.nonzero(THREE_CLASS).shape[0]
                one_number+=ONE_CLASS.shape[0]
                two_number+=TWO_CLASS.shape[0]
                three_number+=THREE_CLASS.shape[0]
                # viewer.line("train/loss",loss.item()*100,ite)
        print(inference_weight,total_count/(one_number+two_number+three_number),one_count/one_number,two_count/two_number,three_count/three_number)








if __name__ == '__main__':
    # fire.Fire()
    main()