# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher LingtengQiu
import fire
import tensorflow as tf
import deepdish as dd
import argparse
import os
import numpy as np
def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3,2,0,1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v

def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights
def transfer(infile,output):
    if output == '':
        output = os.path.splitext(infile)[0] + '.h5'
    outdir = os.path.dirname(output)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    weights = read_ckpt(infile)
    dd.io.save(output, weights)
def load_h5(src):
    pretrained_dict = dd.io.load('./weight.h5')
    keys = pretrained_dict.keys()
    for key,v in pretrained_dict.items():
        print(v.shape)
if __name__ == '__main__':
    fire.Fire()