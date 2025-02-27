import cv2
import torch
import numpy as np
from model import RCNN
from rcnn_dataset import RCNNDataset  

class TextDetector:
    def __init__(self, model_path=None):
        self.model = RCNN(num_classes=2)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def get_region_proposals(self, image):

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rois = ss.process()
        return rois[:2000]  
    
    def prepare_roi(self, image, roi):

        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        roi_image = cv2.resize(roi_image, (224, 224))
        return roi_image
    
    def detect(self, image):

        rois = self.get_region_proposals(image)
        
        roi_batch = []
        original_rois = []
        for roi in rois:
            roi_img = self.prepare_roi(image, roi)
            roi_batch.append(roi_img)
            original_rois.append(roi)
            
        roi_batch = torch.stack([torch.from_numpy(roi) for roi in roi_batch])
        
        with torch.no_grad():
            class_scores, bbox_deltas = self.model(roi_batch, original_rois)
            
        predictions = self.post_process(class_scores, bbox_deltas, original_rois)
        
        return predictions
    
    def post_process(self, class_scores, bbox_deltas, original_rois):

        probs = torch.softmax(class_scores, dim=1)
        
        class_ids = torch.argmax(probs, dim=1)

        final_boxes = self.apply_box_deltas(original_rois, bbox_deltas)

        confident_mask = probs.max(dim=1)[0] > 0.7
        
        return {
            'boxes': final_boxes[confident_mask],
            'labels': class_ids[confident_mask],
            'scores': probs.max(dim=1)[0][confident_mask]
        }
    
    def apply_box_deltas(self, boxes, deltas):
        boxes = torch.tensor(boxes, dtype=torch.float32)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = deltas[:, 0]
        dy = deltas[:, 1]
        dw = deltas[:, 2]
        dh = deltas[:, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = torch.zeros_like(boxes)
        pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes
                