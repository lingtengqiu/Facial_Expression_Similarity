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
    inference_list = os.listdir("./result")
    inference_list = sorted(inference_list)[:-1]
    inference_list = sorted(inference_list)[2:7]
    start = time.time()
    inference_list = [inference_list[-1]]
    if not os.path.exists("./visualize"):
        os.mkdir("./visualize")
    cnt = 0
    for inference_weight in inference_list[::-1]:
        model.load_state_dict(torch.load(os.path.join("./result",inference_weight)))
        model.eval()


        for dataloader in dataloaders:
            for imgs,labels,types,img_ids in tqdm.tqdm(dataloader,desc="dataloader:"):
                types = np.asarray(types)
                lr_sche.step()
                data_time.update(time.time() - start)

                inputs = torch.cat([imgs[0].cuda(),imgs[1].cuda(),imgs[2].cuda()],dim=0)
                with torch.no_grad():
                    features = model(inputs)
                acc,batch_loss = loss_opts.batch_triple_loss_acc(features,labels,types,size_average=True)
                TWO_CLASS = np.nonzero(types=='TWO_CLASS_TRIPLET')[0]
                total = batch_loss[TWO_CLASS].nonzero().view(-1).detach().cpu().numpy()
                index = TWO_CLASS[total].tolist()
                for i in index:
                    a = img_ids[0][i]+'.jpg'
                    b = img_ids[1][i]+'.jpg'
                    c = img_ids[2][i]+".jpg"
                    label = labels[i]
                    img_a = cv2.resize(cv2.imread(a),(160,160))
                    img_b = cv2.resize(cv2.imread(b),(160,160))
                    img_c = cv2.resize(cv2.imread(c),(160,160))
                    img = np.concatenate([img_a,img_b,img_c],axis = 1)
                    cv2.imwrite("{}/{}_{}.jpg".format("./visualize",cnt,label),img)
                    cnt+=1


                # viewer.line("train/loss",loss.item()*100,ite)








if __name__ == '__main__':
    # fire.Fire()
    main()