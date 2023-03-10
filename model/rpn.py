from typing import Dict, List, Optional, Tuple

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops, Conv2dNormActivation, box_iou
from torchvision.models.detection._utils import Matcher, BalancedPositiveNegativeSampler
from torchvision.models.detection.anchor_utils import AnchorGenerator

from utils.utils import BoxCoder, move_dev




class SimpleRPN(nn.Module):
    ''' 
    The simple CNN part of the network. All this does is take in an image (a tensor) of a 
    given size, pass through several convolutional layers stacked with normalisation layers,
    before outputting two sister layers CLS and REG. Which are the objectness scores and 
    the deltas taking anchors to nearest ground truth box.

    Input: [3,256,256]
    Output: [1,anchor_grid_size,anchor_grid_size], [4,anchor_grid_size,anchor_grid_size]
    '''
    
    def __init__(
        self,
        in_channels,
        out_channels,
        num_anchors,
        conv_depth=3
    ):
        super(SimpleRPN, self).__init__()
        convs_list = []
        for _ in range(conv_depth):
            convs_list.append(Conv2dNormActivation(in_channels,in_channels,kernel_size=3,norm_layer=torch.nn.BatchNorm2d))
        self.conv_layers = nn.Sequential(*convs_list)

        self.conv_outchan = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.cls_logits = nn.Conv2d(out_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_deltas = nn.Conv2d(out_channels, num_anchors * 4,kernel_size=1, stride=1)

        # randomly initialise all new layers drawing weights from zero-mean Gaussian (from paper)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, img_tensor):
        logits = []
        bbox_reg = []
        for feature in img_tensor:
            h = self.conv_layers(feature)
            h = self.conv_outchan(h)
            logits.append(self.cls_logits(h)) # more numeric stability without sigmoid
            bbox_reg.append(self.bbox_deltas(h))

        return logits, bbox_reg






class SimplerRPN(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_anchors,
    ):

        super(SimplerRPN, self).__init__()
        self.conv_x = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(out_channels, num_anchors, kernel_size=1, stride=1)
        self.bbx_reg = nn.Conv2d(out_channels, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        logits = []
        bbx_regs = []
        for feature_map in x:
            y = self.conv_x(feature_map)
            logits.append(self.cls_logits(y))
            bbx_regs.append(self.bbx_reg(y))
        return logits, bbx_regs



#investigate the feature maps we're sending to the head, maybe we're just taking the top guy? 
#ie the very first feature map and trying to do everything there?
class SharedConvolutionalLayers(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,48,kernel_size=3,stride=1,padding=1)
        self.conv2 = torch.nn.Conv2d(48,128,kernel_size=3,stride=2,padding=1)
        self.conv3 = torch.nn.Conv2d(128,out_channels,kernel_size=3,stride=2,padding=1)

    def forward(self,x):
        shared_output = []
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))

        shared_output.append(x)

        return shared_output




class SharedConvLayersVGG(nn.Module):
    def __init__(self,out_channels,out_size=256):
        super().__init__()
        vgg16 = torchvision.models.vgg16(weights="VGG16_Weights.DEFAULT").requires_grad_(False)
        modules = list(vgg16.children())[:-2]
        self.vgg16_back = nn.Sequential(*modules)
        module_list = [torch.nn.Conv2d(in_channels=512,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
                       torch.nn.ReLU(inplace=True),
                       torch.nn.AdaptiveAvgPool2d(output_size=(out_size, out_size))]
        self.share_conv_layers = nn.Sequential(*module_list)

    def forward(self,x):
        shared_output = []
        h = self.vgg16_back(x)
        out = self.share_conv_layers(h)
        shared_output.append(out)
        return shared_output







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
        #anchor generator
        sizes,
        aspect_ratios,
        #models
        model,
        in_channels,
        out_channels,
        shared_layers,
        #training arguments
        fg_iou_threshold,
        bg_iou_threshold,
        batch_size_per_image,
        positive_fraction,
        #inference arguments
        pre_nms_top_n,
        nms_threshold,
        post_nms_top_n,
        score_threshold,
        #send to gpu
        device,
    ):
        super().__init__()
        self.device = device
        self.sizes = sizes
        self.aspect_ratios = len(sizes) * aspect_ratios
        #print('sizes',len(sizes),sizes)
        #print('aspect_ratios\n0',len(len(sizes)*aspect_ratios),len(sizes) * aspect_ratios)

        self.anchor_generator = AnchorGenerator(self.sizes, self.aspect_ratios)
        self.num_anchors_per_cell = self.anchor_generator.num_anchors_per_location()[0]
        self.box_coding = BoxCoder(weights=(1.,1.,1.,1.))
        #print('anchor utils method', self.anchor_generator.num_anchors_per_location(),self.anchor_generator.num_anchors_per_location()[0])
        #print('old method (wrong)',len(sizes)*len(aspect_ratios))

        # we need to cover the case where we are loading a model in! in this case it does not need init args
        if isinstance(model,type):
            self.head = model(in_channels, out_channels, self.num_anchors_per_cell)
            self.head.to(self.device)
        else:
            self.head = model
            self.head.to(self.device)

        if isinstance(shared_layers,type):
            self.shared_network = shared_layers(in_channels).to(self.device)
        else:
            self.shared_network = shared_layers
        
        #training
        self.box_similarity = box_iou
        self.proposal_matcher = Matcher(
            fg_iou_threshold,
            bg_iou_threshold,
            allow_low_quality_matches=False
        )
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        #inference
        self._pre_nms_top_n = pre_nms_top_n 
        self.nms_thresh = nms_threshold
        self._post_nms_top_n = post_nms_top_n 
        self.min_box_size = 1e-3
        self.score_thresh = score_threshold

        self.device = device

    def pre_nms_top_n(self):
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self):
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def permute_and_flatten(
        self,
        output,
        B,
        C,
        W,
        H,
    ):
        # turn RPN model output from grid shape to long list (-1 is the number of anchors!)
        # both for 2d cls output and 4d reg output
        # Args
        #   output (Tensor): model output we want to reshape
        #   B (int): batch_size
        #   C (int): number of /output/ channels (actually is 2 for cls, 4 for reg)
        #   W (int): width
        #   H (int): height

        re_output = output.view(B, -1, C, H, W)
        re_output = re_output.permute(0,3,4,1,2)
        return re_output.reshape(B, -1, C)
    
    def concat_box_prediction_layers(
        self,
        objectness,
        pred_bbox_deltas,
    ):
        '''
        This function essentially wraps the premute_and_flatten function for all images in the batch 
        which is how the model will output the objectness/pred_bbox_deltas - in the wrong shape
        
        Output:
            flattened_box_cls_output (Tensor): 
        '''
        flattened_box_cls = []
        flattened_box_reg = []
        for cls_per_level, bbx_reg_per_level in zip(objectness, pred_bbox_deltas):
            B, Axc, H, W = cls_per_level.shape
            _, Ax4, _, _ = bbx_reg_per_level.shape
            A = Ax4 // 4
            C = Axc // A
            flattened_box_cls.append(self.permute_and_flatten(cls_per_level, B, C, W, H))
            flattened_box_reg.append(self.permute_and_flatten(bbx_reg_per_level, B, 4, W, H)) 
        
        box_cls = torch.cat(flattened_box_cls, dim=1).flatten(start_dim=0, end_dim=-2)
        box_reg = torch.cat(flattened_box_reg, dim=1).reshape(-1, 4)
        return box_cls, box_reg


    def _get_top_n_idx(
        self,
        objectness,
        num_anchors_per_level,
    ): 
        # function that will always return the indexes of the boxes most likely to have objects in
        # account for case where there are n > anchors in pre_nms_top_n
        # and when there are numerous anchors per level (in multiple levels)
        top_n_idx = []
        anchor_idx_offset = 0
        for o in  objectness.split(num_anchors_per_level, dim=1):
            anchors_in_level = o.shape[1]
            pre_nms_top_n = min(self.pre_nms_top_n(), anchors_in_level) 
            _, anchor_idx = o.topk(pre_nms_top_n, dim=1)
            top_n_idx.append(anchor_idx + anchor_idx_offset)
            anchor_idx_offset += anchors_in_level
        return torch.cat(top_n_idx, dim=1)  


    def filter_proposals(
        self,
        proposals,
        objectness,
        num_anchors_per_level,
        image_sizes
    ):
        '''
        Import function, tajes decoded Pred_bbox_deltas (called proposals) and their objectness and perform
        selections before and after applying non-maximal suppression to the list of proposals
        Returns on batch level

        Args:
            proposals (Tensor): List of proposals boxes -decoded!
            objectness (Tensor): List of objectness scores from model cls
            num_anchors_per_level (List[int])
            image_sizes (List[Tuple[int,int]])

        '''
        num_images_in_batch = proposals.shape[0]
        objectness.detach() #dont backprop here
        objectness = objectness.reshape(num_images_in_batch,-1) #important: reshape was not acting in place
        #print('objectness.shape',objectness.shape)

        #give anchors in each level an index referring to the level they belong to (should be 1 for us)
        levels = [torch.full((n,), idx, dtype=torch.int64,device=self.device) for idx, n in enumerate(num_anchors_per_level)]
        #print('levels\n',len(levels),'\n',levels)
        levels = torch.cat(levels, 0)
        #print('levels\n',len(levels),'\n',levels)
        levels = levels.reshape(1, -1).expand_as(objectness)


        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level) #pre_nms_top_n happens within
        batch_images = torch.arange(0,num_images_in_batch)
        batch_idx = batch_images[:,None]

        objectness = objectness[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness) # Important!

        final_boxes = []
        final_scores = []
        for boxes, scores, level, image_shape in zip(proposals, objectness_prob, levels, image_sizes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            keep = box_ops.remove_small_boxes(boxes, self.min_box_size)
            boxes, scores, level = boxes[keep], scores[keep], level[keep]
            #print(scores)
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, level = boxes[keep], scores[keep], level[keep]
            #print('scores2',scores)
            keep = box_ops.batched_nms(boxes, scores, level, self.nms_thresh)
            keep = keep[:self.post_nms_top_n()]

            boxes, scores = boxes[keep], scores[keep]
            #print('scores3',scores)
            final_boxes.append(boxes)
            final_scores.append(scores)

        return final_boxes, final_scores


    def assign_targets_to_anchors(
        self,
        anchors,
        targets,
    ):
        '''
        Takes a list of anchors and GT boxes and returns the anchor assignment tensor
        POS:1, NEG:0, NEUTRAL:-1
        and for each anchor the GT box it matches to (if any)
        '''

        labels = []
        matched_gt_boxes = []
        for anchor_per_image, target_per_image in zip(anchors, targets):

            gt_boxes = target_per_image['boxes']
            matched_quality_matrix = self.box_similarity(gt_boxes, anchor_per_image)
            matched_idx = self.proposal_matcher(matched_quality_matrix)
            #clamp because some indices may have -ve values
            matched_gt_boxes_per_img = gt_boxes[matched_idx.clamp(min=0)]

            labels_per_image = matched_idx >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            bg_idx = matched_idx == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_idx] = 0.0

            #discard indices between threshholds
            idx_discard = matched_idx == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[idx_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_img)

        return matched_gt_boxes, labels
    

    def compute_loss(
        self,
        objectness,
        pred_bbox_deltas,
        labels,
        regression_targets,
    ):
        # Function to calculate the loss from cls and reg model outputs
        # Importantly we also use the fg_bg_sampler to only take into account self.batch_size_per_image
        # bbox proposals per calculation
        #across the batch of images we actually turn the fg/bg indices into a really long list sampled_inds (and sampled_[]_inds)
        #the torch.nonzero finds the indices of the elements that are different from 0
        #then both FG and BG contribute to L_CLS but only FG included in L_REG

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds,dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds,dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds,sampled_neg_inds],dim=0)

        objectness = objectness.flatten()
        labels = torch.cat(labels,dim=0)
        regression_targets = torch.cat(regression_targets,dim=0)

        # BCE loss on objectness predictions
        # uses default reduction='mean'
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds],
            labels[sampled_inds],
            reduction='mean',
        )

        # F1 Loss on box parameters (not IoU loss)
        #reduction='sum' here sums up over the batch does not divide by n (ensures roughly same size ass L_cls)
        #ensures loss scale is roughly equal (not dominated by L_cls)
        #also here, as we dont divide by n we are dependant on batch_size
        #discuss 'mean' vs 'sum' reduction!
        #also divided by the number of sampled inds! (256 i think)
        
        box_loss = F.smooth_l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            beta = 1/9,
            reduction='sum'
        ) / (sampled_pos_inds.numel())
        # box_loss = F.l1_loss(
        #     pred_bbox_deltas[sampled_pos_inds],
        #     regression_targets[sampled_pos_inds],
        #     reduction='sum'
        # ) / (sampled_inds.numel())

        # add penalty loss?
        return objectness_loss, box_loss



    def forward(
        self,
        batch_of_images,
        batch_of_anns = None,
    ):
        '''
        Full pass of the RPN algorithm. 
        1. Compute feature maps,  (VGG or homemade)
        2. Get model predictions, 
        3. Generate anchors, 
        4. Re-shape model outputs 
        5. decode model outputs -> proposals,
        6. Filter proposals, 
        7. (If training) Assign targets to anchors, 
        8. (If training) Compute loss, 
        9. Combine losses? back propagate? optimiser step?

        Args:
            batch_of_images (Tensor): images, as tensors, we want to compute predictions for
            batch_of_anns (List[Dict[str,tensor]]): ground-truth boxes, labels, OPTIONAL
        
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one tensor per image
            scores (List[Tensor]): the scores associated to each of the proposal boxes
            losses (Dict[str,Tensor]): the losses from the model in training (empty in test)

        '''

        batch_of_images = move_dev(batch_of_images,self.device)

        feature_maps = self.shared_network(batch_of_images)
        #print('feature maps',len(feature_maps),feature_maps[0].shape)
        
        objectness, pred_bbox_deltas = self.head(feature_maps)
        #print('objectness',objectness[0].detach().cpu().numpy())

        image_shapes = [image.shape[-2:] for image in batch_of_images]
        image_list = ImageList(batch_of_images,image_shapes)
        anchors = self.anchor_generator(image_list,feature_maps)

        num_images_in_batch = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in objectness]

        objectness, pred_bbox_deltas = self.concat_box_prediction_layers(objectness,pred_bbox_deltas)
        #print('objectness2',objectness[0].detach().cpu().numpy())

        proposals = self.box_coding.decode(pred_bbox_deltas.detach(),anchors)
        proposals = proposals.view(num_images_in_batch, -1, 4)
        
        final_boxes, final_scores = self.filter_proposals(proposals, objectness.detach(), num_anchors_per_level, image_shapes)
        #print('\nfinal scores',final_scores[0].cpu().numpy())
        losses = {}
        if self.training or (batch_of_anns is not None):
            if batch_of_anns is None:
                raise ValueError("Targets should not be none in training")

            targets = move_dev(batch_of_anns,self.device)
            matched_gt_boxes, labels = self.assign_targets_to_anchors(anchors,targets)
            regression_targets = self.box_coding.encode(matched_gt_boxes, anchors)
            loss_clf, loss_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_clf": loss_clf, "loss_reg": loss_reg}

        return final_boxes, final_scores, losses





