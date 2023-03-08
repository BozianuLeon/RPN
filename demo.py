import torch
import torchvision

import time
import os
import pickle

from model.rpn import SimpleRPN, SharedConvolutionalLayers, RPNStructure
from utils.dataset import CustomCOCODataset, CustomCOCODataLoader

# Get data
dataset = CustomCOCODataset(root_folder="/home/users/b/bozianu/work/data/val2017",
                            annotation_json="/home/users/b/bozianu/work/data/annotations/instances_val2017.json")
print('Images in dataset:',len(dataset))

train_size = int(0.25 * len(dataset))
val_size = int(0.25 * len(dataset))
test_size = len(dataset) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])


#config
batch_size = 64
n_workers = 2
bbone2rpn_channels = 64
out_channels = 256
sizes = ((16, 32, 64), ) 
#sizes = ((32, ), (64, ), (128, )) #this config is for when multiple feature maps are passed - not the case for us
aspect_ratios = ((0.5, 1.0, 2.0), )
pre_nms_top_n = 40
post_nms_top_n = 40
score_threshold = 0.05
nms_threshold = 0.5
fg_iou_threshold = 0.55
bg_iou_threshold = 0.2
batch_size_per_image = 256
positive_fraction = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\tdevice: {}, num_workers: {}'.format(device,n_workers))


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

    train_init_dataloader = CustomCOCODataLoader(train_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=True)
    val_init_dataloader = CustomCOCODataLoader(val_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=True)
    train_dataloader = train_init_dataloader.loader()
    val_dataloader = val_init_dataloader.loader()

    #training loop
    n_epochs = 1
    factor_C = 1.5
    optimizer = torch.optim.SGD(rpn.head.parameters(), lr=0.01, momentum=0.9)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patence=10,threshold=0.0001,threshold_mode='abs')

    loss_per_epoch = []
    tr_cls_loss, tr_reg_loss = [], []
    val_loss_per_epoch = []
    for epoch in range(n_epochs):
        t_start = time.perf_counter()
        running_loss = 0.0
        running_val_loss = 0.0
        
        for i, data in enumerate(train_dataloader):
            rpn.train()
            
            img,truth = data # Sent to device inside forward pass

            optimizer.zero_grad(set_to_none=True) # Reduce memory operations

            boxes, scores, losses = rpn(img, truth) # Make predictions for this batch and compute losses + gradients

            loss = losses["loss_clf"] + factor_C * losses["loss_reg"]
            loss.backward()  # init backprop
            optimizer.step() # adjust weights
 
            running_loss += loss.detach().item() / len(img) # just for logging, per image loss!

        print('EPOCH: {} \t; TRAIN LOSS: {}'.format(epoch,running_loss), running_loss,len(img))  
        #scheduler.step()
        loss_per_epoch.append(running_loss)

        with torch.no_grad():    
            for j, val_data in enumerate(val_dataloader):
                rpn.eval()
                val_img, val_truth = val_data
                val_boxes, val_scores, val_losses = rpn(val_img, val_truth)
                val_loss = val_losses["loss_clf"] + factor_C * val_losses["loss_reg"]

                running_val_loss += val_loss.detach().item() / len(val_img)
                
            print('EPOCH: {} \t; VAL LOSS: {}'.format(epoch,running_val_loss),running_val_loss, len(val_img))
            val_loss_per_epoch.append(running_val_loss)
        
        t_end = time.perf_counter()
        print('EPOCH duration: {:.3f}s'.format(t_end-t_start))



    path = '/home/users/b/bozianu/work/logs/'+str(time.strftime('%H%M%S'))
    if not os.path.exists(path):
        os.makedirs(path)

    #https://pytorch.org/tutorials/beginner/saving_loading_models.html
    model_save_at = path + "/model-{}e.pth".format(n_epochs)
    # save_at = '/Users/leonbozianu/work/phd/RPN/model/model-{}e.pth'.format(n_epochs)
    torch.save(rpn.head.state_dict(), model_save_at)


    with open(path + "/train-loss-list.pkl", "wb") as fp:
        pickle.dump(loss_per_epoch,fp)

    with open(path + "/val-loss-list.pkl", "wb") as fp:
        pickle.dump(val_loss_per_epoch,fp)

