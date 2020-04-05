from torch import nn


def detect_loss(logits, box_regression, labels, regression_targets):
    pass


def caption_loss(caption_predicts, caption_gt, caption_length):
    pass


class DenseCapRoIHeads(nn.Module):

    def __init__(self,
                 box_describer,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img):

        super(DenseCapRoIHeads, self).__init__()

        self.box_describer = box_describer


    def select_training_samples(self, proposals, targets):
        pass

    def postprocess_detections(self, logits, box_regression, proposals, image_shapes, box_features=None):
        pass

    def forward(self, features, proposals, image_shapes, targets=None):
        pass
