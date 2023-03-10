import torch
import torchvision

import time
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

from model.rpn import SimpleRPN, SharedConvolutionalLayers, SharedConvLayersVGG, RPNStructure
from utils.dataset import CustomCOCODataset, CustomCOCODataLoader


dataset = CustomCOCODataset(root_folder="/home/users/b/bozianu/work/data/val2017",
                            annotation_json="/home/users/b/bozianu/work/data/annotations/instances_val2017.json")
print('Images in dataset:',len(dataset))

train_size = int(0.5 * len(dataset))
val_size = int(0.25 * len(dataset))
test_size = len(dataset) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)
#torch.random.manual_seed(1)

#config
batch_size = 32
n_workers = 2
bbone2rpn_channels = 64
out_channels = 128
sizes = ((16, 32, 64, 128), ) 
aspect_ratios = ((0.5, 1.0, 2.0), )
pre_nms_top_n = {"training": 1000, "testing": 400}
post_nms_top_n = {"training": 100, "testing": 50}
score_threshold = 0.2
nms_threshold = 0.5
fg_iou_threshold = 0.7
bg_iou_threshold = 0.3
batch_size_per_image = 256
positive_fraction = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\tbatch size: {}, num_workers: {}, device: {}'.format(batch_size,n_workers,device))

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])
train_init_dataloader = CustomCOCODataLoader(train_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=True)
val_init_dataloader = CustomCOCODataLoader(val_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=False)
train_dataloader = train_init_dataloader.loader()
val_dataloader = val_init_dataloader.loader()


# //////////////////////////////////// //////////////////////////////////// //////////////////////////////
#for a trained model to load in we must initialise anchor generator outside with same parameters as before
load_model_path = '/home/users/b/bozianu/work/logs/175557/model-20e.pth'
# //////////////////////////////////// //////////////////////////////////// //////////////////////////////

save_at = os.path.dirname(load_model_path)
anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(sizes, aspect_ratios)
num_anchors_per_cell = anchor_generator.num_anchors_per_location()[0]
loaded_model = SimpleRPN(bbone2rpn_channels,out_channels,num_anchors_per_cell)
loaded_model.load_state_dict(torch.load(load_model_path))
loaded_model.eval()
print('\tUsing trained model in eval mode from '+ load_model_path)




rpn_infer = RPNStructure(
    sizes=sizes,
    aspect_ratios=aspect_ratios,
    
    model=loaded_model,
    in_channels=bbone2rpn_channels,
    out_channels=out_channels,
    shared_layers=SharedConvLayersVGG,
    
    fg_iou_threshold=fg_iou_threshold,
    bg_iou_threshold=bg_iou_threshold,
    batch_size_per_image=batch_size_per_image,
    positive_fraction=positive_fraction,
    
    pre_nms_top_n=pre_nms_top_n,
    nms_threshold=nms_threshold,
    post_nms_top_n=post_nms_top_n,
    score_threshold=score_threshold,

    device=device
)




index,(img,truth) = next(enumerate(train_dataloader))
rpn_infer.eval()
boxes1, scores1, losses1 = rpn_infer(img)
print(scores1) #score threshold is killing us here!


