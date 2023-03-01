import torch
import torchvision
from tqdm.auto import tqdm

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import pickle

from model.rpn import SimpleRPN, RPNStructure, SimplerRPN, SharedConvolutionalLayers
from utils.dataset import CustomCOCODataset, CustomCOCODataLoader
from utils.utils import BoxCoder



dataset = CustomCOCODataset(root_folder="data/val2017",
                            annotation_json="data/annotations/instances_val2017.json")
print('Images in dataset:',len(dataset))



batch_size = 45

train_size = int(0.3 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size -val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])





#config
bbone2rpn_channels = 64
out_channels = 256
sizes = ((16, 32, 64), ) 
#sizes = ((32, ), (64, ), (128, )) #this config is for when multiple feature maps are passed - not the case for us
aspect_ratios = ((0.5, 1.0, 2.0), )
pre_nms_top_n = 40
post_nms_top_n = 40
score_threshold = 0.05
nms_threshold = 0.5
fg_iou_threshold = 0.45
bg_iou_threshold = 0.2
batch_size_per_image = 256
positive_fraction = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ',device)


rpn = RPNStructure(
    #anchors
    sizes=sizes,
    aspect_ratios=aspect_ratios,
    #training
    model=SimpleRPN,
    in_channels=bbone2rpn_channels,
    out_channels=out_channels,
    shared_layers=SharedConvolutionalLayers,
    #loss calculating
    fg_iou_threshold=fg_iou_threshold,
    bg_iou_threshold=bg_iou_threshold,
    batch_size_per_image=batch_size_per_image,
    positive_fraction=positive_fraction,
    #filtering proposals
    pre_nms_top_n=pre_nms_top_n,
    nms_threshold=nms_threshold,
    post_nms_top_n=post_nms_top_n,
    score_threshold=score_threshold,
    #gpus
    device=device,
)




if __name__=='__main__':

    train_init_dataloader = CustomCOCODataLoader(train_dataset,batch_size,num_workers=0,shuffle=True)
    val_init_dataloader = CustomCOCODataLoader(val_dataset,batch_size,num_workers=0,shuffle=True)
    train_dataloader = train_init_dataloader.loader()
    val_dataloader = val_init_dataloader.loader()

    #training loop
    n_epochs = 10
    factor_C = 1.5
    optimizer = torch.optim.SGD(rpn.head.parameters(), lr=0.001, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patence=10,threshold=0.0001,threshold_mode='abs')

    loss_per_epoch = []
    val_loss_per_epoch = []
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_val_loss = 0.0
        
        for i, data in tqdm(enumerate(train_dataloader),total=len(train_dataloader)):
            rpn.train()

            img,truth = data # Sent to device inside forward pass

            optimizer.zero_grad(set_to_none=True) # Reduce memory operations

            boxes, scores, losses = rpn(img, truth) # Make predictions for this batch and compute losses + gradients

            loss = losses["loss_clf"] + factor_C * losses["loss_reg"]
            loss.backward()  # init backprop
            optimizer.step() # adjust weights

            running_loss += loss.item()
        print('TRAIN LOSS:',running_loss/len(train_dataloader))    
        #scheduler.step()
        loss_per_epoch.append(running_loss/len(train_dataloader))

        with torch.no_grad():    
            for j, val_data in enumerate(val_dataloader):
                rpn.eval()
                val_img, val_truth = val_data
                val_boxes, val_scores, val_losses = rpn(val_img, val_truth)
                val_loss = val_losses["loss_clf"] + factor_C * val_losses["loss_reg"]

                running_val_loss += val_loss.item()
                
            print('VAL LOSS',running_val_loss/len(val_dataloader))
            val_loss_per_epoch.append(running_val_loss/len(val_dataloader))


        #https://pytorch.org/tutorials/beginner/saving_loading_models.html
        save_at = '/Users/leonbozianu/work/phd/RPN/model/model-{}e.pth'.format(n_epochs)
        torch.save(rpn.head.state_dict(), save_at)


        with open("train-loss-list.pkl", "wb") as fp:
            pickle.dump(loss_per_epoch,fp)

        with open("val-loss-list.pkl", "wb") as fp:
            pickle.dump(val_loss_per_epoch,fp)

