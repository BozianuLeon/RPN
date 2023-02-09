import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import boxes as box_ops, Conv2dNormActivation, box_iou
from torchvision.models.detection._utils import Matcher, BoxCoder, BalancedPositiveNegativeSampler
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






class RPNStructure(nn.Module):
    '''
    Contains the functions and logic needed to run the RPN. The forward call does the heavy lifting
    here. Follows the torchvision documentation closely, but does not use anchors per level.
    Args:
        anchor_generator: torchvision native module
        model: the convolutional neural network 
        fg_iou_thresh: IOU score necessary to become a positive anchor example (FG) in training
        bg_iou_thresh: IOU score below which anchor is considered negative (BG) in training
        batch_size_per_image: numb of anchors used in the compute loss function
        pos_fraction: proportion of anchors in compute loss that should be positive
        pre_nms_top_n: number of top-scoring proposals we keep before NMS
        nms_thresh: threshold used in NMS [0,1]
        post_nms_top_n: number of top-scoring proposals we keep after NMS

    Returns:
        Boxes, objectness_scores, losses
    '''

    def __init__(
        self,
        model,
        #training arguments
        fg_iou_threshold,
        bg_iou_threshold,
        batch_size_per_image,
        positive_fraction,
        #inference arguments
        pre_nms_top_n,
        nms_threshold,
        post_nms_top_n,
        score_threshold
    ):
        super().__init__()
        self.head = model
        self.anchor_generator = AnchorGenerator
        self.box_coding = BoxCoder(weights=(1.,1.,1.,1.))
        
        #training
        self.box_similarity = box_iou
        self.proposal_matcher = Matcher(
            fg_iou_threshold,
            bg_iou_threshold,
            allow_low_quality_matches=True
        )
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        #inference
        self._pre_nms_top_n = pre_nms_top_n #for now just one topN for both train+test
        self.nms_thresh = nms_threshold
        self._post_nms_top_n = post_nms_top_n #for now just one topN for both train+test
        self.score_thresh = score_threshold
        self.min_size = 1e-3




    #....

    def foward(
        self,
        batch_of_images,
        batch_of_anns,
    ):
        '''
        Full pass of the RPN algorithm. 1. Compute feature maps, 2. Get model predictions, 
        3. Generate anchors, 4. Re-shape model outputs 5. decode model outputs -> proposals,
        6. Filter proposals, 7. (If training) Assign targets to anchors, 
        8. (If training) Compute loss, 9. Combine losses? back propagate? optimiser step?

        Args:
            batch_of_images (ImageList): images, as tensors, we want to compute predictions for
            batch_of_anns (List[Dict[str,tensor]]): ground-truth boxes, labels, optional
        
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one tensor per image
            scores (List[Tensor]): the scores associated to each of the proposal boxes
            losses (Dict[str,Tensor]): the losses from the model in training (empty in test)

        '''

        feature_maps = self.shared_network(batch_of_images)
        
        objectness, pred_bbox_deltas = self.head(feature_maps)
        anchors = self.anchor_generator(batch_of_images,feature_maps)

        num_images_in_batch = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0]*s[1]*s[2] for s in num_anchors_per_level_shape_tensors]

        objectness, pred_bbox_deltas = self.concat_box_prediction_layers(objectness,pred_bbox_deltas)
        proposals = self.box_coding.decode(pred_bbox_deltas.detach(),anchors)
        proposals = proposals.view(num_images_in_batch, -1, 4)
        final_boxes, final_scores = self.filter_proposals(proposals, objectness, batch_of_images, num_anchors_per_level)

        losses = {}
        if self.training:
            print("Trainning RPN ...")
            if batch_of_anns is None:
                raise ValueError("Targets should not be none in training")

            targets = batch_of_anns[] #how are we inputting truth
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors,targets)
            regression_targets = self.box_coding.encode(matched_gt_boxes, anchors)
            loss_clf, loss_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_clf": loss_clf, "loss_reg": loss_reg}

        return final_boxes, final_scores, losses





