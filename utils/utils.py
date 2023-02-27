import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.functional import Tensor
from torchvision.ops import boxes as box_ops
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict





class EarlyStopper:
    def __init__(self,
                 patience=1,
                 min_delta=0
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf
    

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False





@torch.jit.script_if_tracing
def encode_boxes(reference_boxes: Tensor, proposals: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets




class BoxCoder:
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """
    def __init__(self, weights:Tuple[float, float, float, float]=(1.0, 1.0, 1.0, 1.0), bbx_clip:float=math.log(1000.0 / 16) ) -> None:
        self.weights = weights
        self.bbx_clip = bbx_clip

    def decode(self, pred_bbx_deltas:Tensor, anchors:List[Tensor]):
        boxes_per_image  = [ a.size(0) for a in anchors ]
        # print('boxes_per_image:',boxes_per_image)
        # print('boxes_sum',sum(boxes_per_image))
        boxes_sum = sum(boxes_per_image)
        concat_anchors = torch.cat(anchors, dim=0)
        # print('concat_anchors:',len(concat_anchors))
        # print('pred_bbx_deltas',pred_bbx_deltas.shape)
        assert boxes_sum > 0, "Sum of pred boxes per image in decode < 0"
        pred_bbx = pred_bbx_deltas.reshape(boxes_sum, -1)
        pred_bbx = self.decode_single(pred_bbx, concat_anchors)
        return pred_bbx.reshape(boxes_sum, -1, 4)


    def decode_single(self, pred_bbx_deltas:Tensor, anchors:Tensor) -> Tensor:
        """
        Notes:
        0 or 0::4 = x1, 1 or 1::4 = y1, 2 or 2::4 =x2, 3 or 3::4 =y2
        """
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights

        dx = pred_bbx_deltas[:, 0::4] / wx
        dy = pred_bbx_deltas[:, 1::4] / wy
        dw = pred_bbx_deltas[:, 2::4] / ww
        dh = pred_bbx_deltas[:, 3::4] / wh

        #Avoid sending large values to exp
        dw = dw.clamp(max=self.bbx_clip)
        dh = dh.clamp(max=self.bbx_clip)

        pred_cx = dx * widths[:, None] + cx[:, None]
        pred_cy = dy * heights[:, None] + cy[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_cx.dtype) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_cy.dtype) * pred_w

        pred_boxes1 = pred_cx - c_to_c_w
        pred_boxes2 = pred_cy - c_to_c_h
        pred_boxes3 = pred_cx + c_to_c_w
        pred_boxes4 = pred_cy + c_to_c_h
        pred_boxes = torch.stack([pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4], dim=2).flatten(1)

        return pred_boxes

    def encode(self, matched_gt_boxes:List[Tensor], proposed_boxes:List[Tensor]) -> List[Tensor]:
        boxes_per_image = [len(box) for box in proposed_boxes]
        reference_boxes = torch.cat(matched_gt_boxes, dim=0)
        proposals = torch.cat(proposed_boxes, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, dim=0)

    def encode_single(self, reference_boxes:Tensor, proposals:Tensor) -> Tensor:
        device = reference_boxes.device
        dtype = reference_boxes.dtype
        weights = torch.as_tensor(self.weights, device=device, dtype=dtype)
        targets = encode_boxes(reference_boxes, proposals, weights)
        return targets




