import torch
import torchvision

import time
import os
import pickle
import matplotlib.pyplot as plt

from model.rpn import SimpleRPN, SharedConvolutionalLayers, SharedConvLayersVGG, RPNStructure
from utils.dataset import CustomCOCODataset, CustomCOCODataLoader

# Get data
dataset = CustomCOCODataset(root_folder="/home/users/b/bozianu/work/data/train2017",
                            annotation_json="/home/users/b/bozianu/work/data/annotations/instances_train2017.json")
print('Images in dataset:',len(dataset))

train_size = int(0.05 * len(dataset))
val_size = int(0.025 * len(dataset))
test_size = len(dataset) - train_size - val_size
print('\ttrain / val / test size : ',train_size,'/',val_size,'/',test_size)

torch.random.manual_seed(1)
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,val_size,test_size])


#config
batch_size = 16
n_workers = 2
bbone2rpn_channels = 64
out_channels = 128
sizes = ((16, 32, 64), ) 
#sizes = ((32, ), (64, ), (128, )) #this config is for when multiple feature maps are passed - not the case for us
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


rpn = RPNStructure(
    #anchors
    sizes=sizes,
    aspect_ratios=aspect_ratios,
    #training
    model=SimpleRPN,
    in_channels=bbone2rpn_channels,
    out_channels=out_channels,
    shared_layers=SharedConvLayersVGG,
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
    val_init_dataloader = CustomCOCODataLoader(val_dataset,batch_size,num_workers=n_workers,shuffle=True,drop_last=False)
    train_dataloader = train_init_dataloader.loader()
    val_dataloader = val_init_dataloader.loader()

    #training loop
    n_epochs = 15
    factor_C = 0.35#1.1
  
    #optimizer = torch.optim.SGD(rpn.head.parameters(), lr=0.01, momentum=0.9)
    params = list(rpn.head.parameters()) + list(rpn.shared_network.parameters())
    optimizer = torch.optim.Adam(params, lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=5,threshold=0.0001,threshold_mode='abs') #https://hasty.ai/docs/mp-wiki/scheduler/reducelronplateau

    loss_per_epoch = []
    tr_cls_loss, tr_reg_loss = [], []
    val_loss_per_epoch = []
    for epoch in range(n_epochs):
        t_start = time.perf_counter()
        running_loss = 0.0
        tr_l_cls, tr_l_reg = 0.0, 0.0
        running_val_loss = 0.0
        
        for i, data in enumerate(train_dataloader):
            rpn.train()
            
            img,truth = data # Sent to device inside forward pass

            optimizer.zero_grad(set_to_none=True) # Reduce memory operations

            boxes, scores, losses = rpn(img, truth) # Make predictions for this batch and compute losses + gradients
            #print('LOSSES',losses["loss_clf"].item(),losses["loss_reg"].item())
            loss = losses["loss_clf"] + factor_C * losses["loss_reg"]
            loss.backward()  # init backprop
            optimizer.step() # adjust weights
 
            running_loss += loss.detach().item()  # just for logging ~dependent on batch_size (for both tr and val)
            tr_l_cls += losses["loss_clf"].detach().item()
            tr_l_reg += losses["loss_reg"].detach().item()

        print('EPOCH: {} \t; TRAIN LOSS: {}'.format(epoch,running_loss/len(train_dataloader)))  
        
        loss_per_epoch.append(running_loss/len(train_dataloader))
        tr_cls_loss.append(tr_l_cls/len(train_dataloader))
        tr_reg_loss.append(factor_C*tr_l_reg/len(train_dataloader))

        with torch.no_grad():    
            for j, val_data in enumerate(val_dataloader):
                rpn.eval()
                val_img, val_truth = val_data
                val_boxes, val_scores, val_losses = rpn(val_img, val_truth)
                val_loss = val_losses["loss_clf"] + factor_C * val_losses["loss_reg"]

                running_val_loss += val_loss.detach().item() 

            print('EPOCH: {} \t; VAL LOSS: {}'.format(epoch,running_val_loss/len(val_dataloader)))
            scheduler.step(running_val_loss/len(val_dataloader))
            val_loss_per_epoch.append(running_val_loss/len(val_dataloader))
        
        t_end = time.perf_counter()
        print('EPOCH duration: {:.3f}s'.format(t_end-t_start))
        del img, truth, val_img, val_truth


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


plt.figure()
x_axis = torch.arange(n_epochs)
plt.plot(x_axis,loss_per_epoch,'--',label='training loss')
plt.plot(x_axis,val_loss_per_epoch,'--',label='val loss')
plt.plot(x_axis,tr_cls_loss,'--',label='training cls loss')
plt.plot(x_axis,tr_reg_loss,'--',label='(Scaled) training reg loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Arbitrary Loss')
plt.savefig(path+'/losses.png')


