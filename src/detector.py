import torch
import torchvision
import cv2
import numpy as np


class ObjectDetector:
    """استفاده از Faster R-CNN pretrained برای تشخیص اشیاء"""
    def __init__(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def detect(self, frame):
        """تشخیص اشیاء: بازگشت (boxes, scores)"""
        img_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)

        boxes = predictions[0]["boxes"].cpu().numpy()
        scores = predictions[0]["scores"].cpu().numpy()

        # فیلتر کردن با confidence threshold
        mask = scores > 0.5
        return boxes[mask], scores[mask]
