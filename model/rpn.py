import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation, box_iou
from torchvision.models.detection._utils import Matcher, BoxCoder
from torchvision.models.detection.anchor_utils import AnchorGenerator





class SimpleRPN(nn.module):
    def __init__(
        self,
        in_channels,
        num_anchors,
        conv_depth
    ):
        ''' 
        The simple CNN part of the network. All this does is take in an image (a tensor) of a 
        given size, pass through several convolutional layers stacked with normalisation layers,
        before outputting two sister layers CLS and REG. Which are the objectness scores and 
        the deltas taking anchors to nearest ground truth box.

        Input: [3,256,256]
        Output: [1,anchor_grid_size,anchor_grid_size], [4,anchor_grid_size,anchor_grid_size]
        '''
        convs_list = []
        for _ in range(conv_depth):
            convs_list.append(Conv2dNormActivation(in_channels,in_channels,kernel_size=3,norm_layer=torch.nn.BatchNorm2d))
        self.conv_layers = nn.Sequential(*convs_list)

        self.sigmoid = nn.Sigmoid()
        self.cls_logits = nn.Conv2d(in_channels,num_anchors,kernel_size=1,stride=1)
        self.bbox_deltas = nn.Conv2d(in_channels,num_anchors*4,kernel_size=1,stride=1)

    def forward(self, img_tensor):
        h = self.conv_layers(img_tensor)

        logits = self.sigmoid(self.cls_logits(h))
        bbox_deltas = self.bbox_deltas(h)

        return logits, bbox_deltas









