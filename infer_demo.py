import torch
import torchvision

import time
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt

from model.rpn import SimpleRPN, SharedConvolutionalLayers, SharedConvLayersVGG,RPNStructure
from utils.dataset import CustomCOCODataset, CustomCOCODataLoader



dataset = CustomCOCODataset(root_folder="/home/users/b/bozianu/work/data/val2017",
                            annotation_json="/home/users/b/bozianu/work/data/annotations/instances_val2017.json")
print('Images in dataset:',len(dataset))

train_size = int(0.5 * len(dataset))
val_size = int(0.25 * len(dataset))
test_size = len(dataset) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])

#config
batch_size = 12
n_workers = 2
bbone2rpn_channels = 64
out_channels = 256
sizes = ((16, 32, 128), ) 
#sizes = ((32, ), (64, ), (128, )) #this config is for when multiple feature maps are passed - not the case for us
aspect_ratios = ((0.5, 1.0, 2.0), )
pre_nms_top_n = 40
post_nms_top_n = 40
score_threshold = 0.2
nms_threshold = 0.5
fg_iou_threshold = 0.55
bg_iou_threshold = 0.2
batch_size_per_image = 256
positive_fraction = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\tbatch size: {}, num_workers: {}, device: {}'.format(batch_size,n_workers,device))

train_init_dataloader = CustomCOCODataLoader(train_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=True)
val_init_dataloader = CustomCOCODataLoader(val_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=True)
train_dataloader = train_init_dataloader.loader()
val_dataloader = val_init_dataloader.loader()


#for a trained model to load in we must initialise anchor generator outside
# with same parameters as before
load_model_path = '/home/users/b/bozianu/work/logs/130153/model-50e.pth'
save_at = os.path.dirname(load_model_path)
anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(sizes, aspect_ratios)
num_anchors_per_cell = anchor_generator.num_anchors_per_location()[0]
loaded_model = SimpleRPN(bbone2rpn_channels,out_channels,num_anchors_per_cell)
loaded_model.load_state_dict(torch.load(load_model_path))
loaded_model.eval()
print('\tUsing trained model in eval mode from '+ load_model_path)


rpn = RPNStructure(
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




index,(img,truth) = next(enumerate(val_dataloader))
rpn.eval()
boxes1, scores1, losses1 = rpn(img)

for i in range(len(img)):
    boxes_pred = boxes1[i]
    scores_pred = scores1[i]

    fig,ax = plt.subplots()
    for bbx in truth[i]["boxes"]:
        x,y=float(bbx[0]),float(bbx[1])
        w,h=float(bbx[2])-float(bbx[0]),float(bbx[3])-float(bbx[1])
        
        bb = matplotlib.patches.Rectangle((x,y),w,h,lw=2,ec='limegreen',fc='none')
        ax.add_patch(bb)
        
    for boox, scoore in zip(boxes_pred,scores_pred):
        print(boox,scoore)
        x_pred,y_pred=float(boox[0]),float(boox[1])
        w_pred,h_pred=float(boox[2])-float(boox[0]),float(boox[3])-float(boox[1])
        
        bb_pred = matplotlib.patches.Rectangle((x_pred,y_pred),w_pred,h_pred,lw=2,ec='red',fc='none')
        ax.add_patch(bb_pred)
        
    image = torch.permute(img[i], (1, 2, 0))
    ax.imshow(image)

    if not os.path.exists(save_at+'/images/'):
        os.makedirs(save_at+'/images/')

    fig.savefig(save_at+'/images/infer_test{}.png'.format(str(i)))
    plt.close()


















