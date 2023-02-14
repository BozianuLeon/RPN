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

        logits = self.cls_logits(h) #more numeric stability without sigmoid
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
        score_threshold
    ):
        super().__init__()
        self.head = model
        self.shared_network = shared_layers
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
            flattened_box_cls.append(self.permute_and_reshape(cls_per_level, B, C, W, H))
            flattened_box_reg.append(self.permute_and_reshape(bbx_reg_per_level, B, 4, W, H)) 
        
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
        objectness.reshape(num_images_in_batch,-1)

        #give anchors in each level an index referring to the level they belong to (should be 1 for us)
        levels = [torch.full((anchors_in_level,), idx, dtype=torch.int64) for idx, anchors_in_level in enumerate(num_anchors_per_level)]
        levels = torch.cat(levels,dim=0)
        levels = levels.reshape(1,-1).expand_as(objectness)

        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level) #pre_nms_top_n happens within
        batch_images = torch.arange(0,num_images_in_batch)
        batch_idx = batch_images[:,None]

        objectness = objectness[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]

        final_boxes = []
        final_scores = []
        for boxes, scores, level, image_shape in zip(proposals, objectness, levels, image_sizes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            keep = box_ops.remove_small_boxes(boxes, self.min_box_size)
            boxes, scores, level = boxes[keep], scores[keep], level[keep]

            keep = torch.where(scores >= self.score_threshold)[0]
            boxes, scores, level = boxes[keep], scores[keep], level[keep]

            keep = box_ops.batched_nms(boxes, scores, level, self.nms_theshold)
            keep = keep[:self.post_nms_top_n]

            boxes, scores = boxes[keep], scores[keep]

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
        # Function to simultaneously calculate the loss from cls and reg model outputs
        # Importantly we also use the fg_bg_sampler to only take into account self.batch_size_per_image
        # bbox proposals per calculation

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds,dim=0)).squeeze(1)
        sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds,dim=0)).squeeze(1)

        sampled_inds = torch.cat([sampled_pos_inds,sampled_neg_inds],dim=0)
        
        objectness = objectness.flatten()
        labels = torch.cat(labels,dim=0)
        regression_targets = torch.cat(regression_targets,dim=0)

        # BCE loss on objectness predictions
        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness[sampled_inds],
            labels[sampled_inds],
        )

        # F1 Loss on box parameters (not IoU loss)
        box_loss = F.l1_loss(
            pred_bbox_deltas[sampled_pos_inds],
            regression_targets[sampled_pos_inds],
            reduction='sum'
        ) / (sampled_inds.numel())

        return objectness_loss, box_loss


    #....

    def foward(
        self,
        batch_of_images,
        batch_of_anns,
    ):
        '''
        Full pass of the RPN algorithm. 
        1. Compute feature maps, 
        2. Get model predictions, 
        3. Generate anchors, 
        4. Re-shape model outputs 
        5. decode model outputs -> proposals,
        6. Filter proposals, 
        7. (If training) Assign targets to anchors, 
        8. (If training) Compute loss, 
        9. Combine losses? back propagate? optimiser step?

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
        final_boxes, final_scores = self.filter_proposals(proposals, objectness, num_anchors_per_level, batch_of_images.image_sizes)

        losses = {}
        if self.training:
            print("Trainning RPN ...")
            if batch_of_anns is None:
                raise ValueError("Targets should not be none in training")

            targets = batch_of_anns[''] #how are we inputting truth
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors,targets)
            regression_targets = self.box_coding.encode(matched_gt_boxes, anchors)
            loss_clf, loss_reg = self.compute_loss(objectness, pred_bbox_deltas, labels, regression_targets)
            losses = {"loss_clf": loss_clf, "loss_reg": loss_reg}

        return final_boxes, final_scores, losses





