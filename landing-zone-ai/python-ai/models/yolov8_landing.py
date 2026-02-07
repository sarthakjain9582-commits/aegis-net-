"""
YOLOv8-Nano based segmentation model for UAV Landing Zone Detection
Uses Ultralytics YOLOv8 with custom segmentation head
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np

class YOLOv8LandingZone(nn.Module):
    """
    YOLOv8-Nano backbone with custom segmentation head for binary landing zone detection.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load YOLOv8-nano segmentation model as backbone
        self.yolo = YOLO('yolov8n-seg.pt' if pretrained else 'yolov8n-seg.yaml')
        
        # Custom segmentation head for binary output
        # YOLOv8n-seg outputs feature maps we can use
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),  # MC Dropout
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5),  # MC Dropout
            nn.Conv2d(32, 1, kernel_size=1),  # Binary output
            nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        )
        
    def forward(self, x):
        """
        Forward pass returning segmentation mask.
        """
        # Get YOLO backbone features
        # Using the model's backbone directly
        features = self._extract_features(x)
        
        # Apply custom segmentation head
        mask = self.seg_head(features)
        return mask
    
    def _extract_features(self, x):
        """Extract features from YOLOv8 backbone"""
        # Use YOLO's model backbone
        model = self.yolo.model
        
        # Forward through backbone
        for i, m in enumerate(model.model):
            x = m(x)
            if i == 9:  # After P3 features (early enough for fine details)
                break
        return x
    
    def freeze_backbone(self):
        """Freeze YOLOv8 backbone layers"""
        for param in self.yolo.model.parameters():
            param.requires_grad = False
        print("YOLOv8 backbone frozen.")


class YOLOv8SegmentationWrapper:
    """
    Wrapper for using YOLOv8 segmentation directly (simpler approach).
    Uses YOLOv8's built-in segmentation and converts to binary landing zone mask.
    """
    def __init__(self, model_path='yolov8n-seg.pt'):
        self.model = YOLO(model_path)
        self.safe_classes = ['road', 'pavement', 'ground', 'field']  # Classes considered safe
        
    def predict(self, image, conf_threshold=0.25):
        """
        Run segmentation and return binary safe/unsafe mask.
        """
        results = self.model(image, conf=conf_threshold)
        
        if len(results) == 0 or results[0].masks is None:
            # No detections - return uniform low confidence
            h, w = image.shape[:2] if hasattr(image, 'shape') else (256, 256)
            return np.zeros((h, w), dtype=np.float32)
        
        # Get masks and classes
        masks = results[0].masks.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        names = results[0].names
        
        # Create safe zone mask
        h, w = masks.shape[1:3]
        safe_mask = np.zeros((h, w), dtype=np.float32)
        
        for i, (mask, cls_id) in enumerate(zip(masks, classes)):
            class_name = names[int(cls_id)].lower()
            if any(safe in class_name for safe in self.safe_classes):
                safe_mask = np.maximum(safe_mask, mask)
        
        return safe_mask
    
    def predict_with_uncertainty(self, image, n_passes=10):
        """
        MC Dropout-style uncertainty by running augmented predictions.
        Since YOLO doesn't have dropout in inference, we use augmentation.
        """
        predictions = []
        
        for _ in range(n_passes):
            # Slight augmentation for variation
            aug_image = self._augment(image)
            mask = self.predict(aug_image)
            predictions.append(mask)
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        return mean_pred, variance
    
    def _augment(self, image):
        """Apply slight random augmentation"""
        import cv2
        # Small random brightness/contrast adjustment
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-10, 10)
        return np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
