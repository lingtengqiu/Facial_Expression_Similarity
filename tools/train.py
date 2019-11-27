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
import cv2
import numpy as np
import tqdm
from utils import loss_opts
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from utils.board import Visualizer
from utils.distributed import AverageMeter
import time
from utils.init_func import group_weight

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
        # for key,val in model.module.backbone.named_parameters():
        #     val.requires_grad = False


    #model lr method
    # params_list = []
    # params_list = group_weight(params_list, model.module.backbone,
    #                            nn.BatchNorm2d, cfg.SOLVER.BASE_LR/10)
    # for module in model.module.business:
    #     params_list = group_weight(params_list, module, nn.BatchNorm2d,
    #                                cfg.SOLVER.BASE_LR)


    batch_time = AverageMeter()
    data_time = AverageMeter()
    #optimizer
    optimizer = getattr(torch.optim,cfg.SOLVER.OPTIM)(model.parameters(),lr = cfg.SOLVER.BASE_LR,weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    lr_sche = torch.optim.lr_scheduler.MultiStepLR(optimizer,cfg.SOLVER.STEPS,gamma= cfg.SOLVER.GAMMA)
    #dataset
    datasets  = make_dataset(cfg)
    dataloaders = make_dataloaders(cfg,datasets,True)
    iter_epoch = (cfg.SOLVER.MAX_ITER)//len(dataloaders[0])+1
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    ite = 0
    batch_it = [i *cfg.SOLVER.IMS_PER_BATCH for i in range(1,4)]


    # start time
    model.train()
    start = time.time()
    for epoch in tqdm.tqdm(range(iter_epoch),desc="epoch"):
        for dataloader in dataloaders:
            for imgs,labels,types in tqdm.tqdm(dataloader,desc="dataloader:"):
                lr_sche.step()
                data_time.update(time.time() - start)

                inputs = torch.cat([imgs[0].cuda(),imgs[1].cuda(),imgs[2].cuda()],dim=0)
                features = model(inputs)
                acc,loss = loss_opts.batch_triple_loss(features,labels,types,size_average=True)
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                ite+=1
                # viewer.line("train/loss",loss.item()*100,ite)
                print(acc,loss)
                batch_time.update(time.time() - start)
                start = time.time()

                print('Epoch: [{0}][{1}/{2}]\n'
                      'Time: {data_time.avg:.4f} ({batch_time.avg:.4f})\n'.format(
                    epoch,ite, len(dataloader),
                    data_time=data_time, batch_time=batch_time),
                    flush=True)

        torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR,"{}_{}.pth".format(cfg.MODEL.META_ARCHITECTURE,epoch)))








if __name__ == '__main__':
    main()
