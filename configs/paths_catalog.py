# Copyright (c) Sensetime, Inc. All Rights Reserved.
# Author: Researcher LingtengQiu

import os


class DatasetCatalog(object):
    DATA_DIR = "../"
    DATASETS = {
        "FEC_TRAIN": {
            "img_dir": "FEC/train_align",
            "ann_file": "FEC/train_list"
        },
        "FEC_TEST": {
            "img_dir": "FEC/test_rotation",
            "ann_file":"FEC/test_list"
        },
        "FEC_JIAQI_TRAIN": {
            "img_dir": "FEC/train_align_face1",
            "ann_file": "FEC/jiaqi_align_train.txt"
        },
        "FEC_JIAQI_TEST": {
            "img_dir": "FEC/test_align_face",
            "ann_file": "FEC/jiaqi_align_test.txt"
        },
        "FEC_YIYANG_TEST": {
            "img_dir": "FEC/test_sim",
            "ann_file": "FEC/jiaqi_test.txt"
        },
    }

    @staticmethod
    def get(name):
        if "FEC" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="FEC_Dataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


