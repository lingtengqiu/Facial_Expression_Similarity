from models.inception_N22 import inception_NN2
from models.inception_resnet import inception_resnet_v1
from models.Fee_net import Fee_net
from models.res_net18 import resnet18
model_map = {"inception_N22":inception_NN2
             ,"inception_resnet_v1":inception_resnet_v1,
             "resnet18":resnet18}
in_channel_map = {"inception_N22":1024
             ,"inception_resnet_v1":1792,
                  "resnet18":512}
avg_pool_size = {"inception_N22":7
             ,"inception_resnet_v1":5,
                 "resnet18":7}
def build_model(cfg):
    return Fee_net(model_map[cfg.MODEL.META_ARCHITECTURE],in_channel_map[cfg.MODEL.META_ARCHITECTURE],avg_pool_size[cfg.MODEL.META_ARCHITECTURE],cfg.MODEL.DENSENET)
