import os
import torch
import numpy as np
from .preprocessing import preprocess_image
from .heatmap import generate_heatmap
from config import Config
from models.yolov8_landing import YOLOv8LandingZone, YOLOv8SegmentationWrapper

class InferenceService:
    def __init__(self, use_wrapper=True):
        """
        Initialize inference service.
        
        Args:
            use_wrapper: If True, use YOLO's built-in segmentation (simpler).
                        If False, use custom YOLOv8LandingZone model.
        """
        self.use_wrapper = use_wrapper
        
        if use_wrapper:
            # Use YOLOv8 segmentation directly
            self.model = YOLOv8SegmentationWrapper()
            print("YOLOv8 Segmentation Wrapper loaded.")
        else:
            # Use custom model with trained weights
            self.model = YOLOv8LandingZone(pretrained=True)
            model_path = Config.MODEL_PATH.replace('landing_model.pth', 'yolo_landing.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                print("Custom YOLOv8 model loaded.")
            else:
                print("Warning: Custom weights not found, using pretrained.")
            self.model.eval()

    def predict(self, image_file, mc_samples=10):
        """
        Runs inference with uncertainty estimation.
        Works with both YOLOv8 wrapper and custom model.
        """
        from PIL import Image
        import cv2
        
        # Load image
        if isinstance(image_file, str):
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif hasattr(image_file, 'read'):
            image = np.array(Image.open(image_file))
        else:
            image = np.array(image_file)
        
        if self.use_wrapper:
            # Use YOLO wrapper's built-in uncertainty
            mean_pred, variance = self.model.predict_with_uncertainty(image, n_passes=mc_samples)
        else:
            # Use custom model with MC Dropout
            input_tensor = preprocess_image(image_file)
            predictions = []
            self.model.train()  # Enable Dropout
            
            with torch.no_grad():
                for _ in range(mc_samples):
                    output = self.model(input_tensor)
                    output = torch.sigmoid(output)
                    predictions.append(output.cpu().numpy())
            
            predictions = np.array(predictions)
            mean_pred = np.mean(predictions, axis=0)[0, 0]
            variance = np.var(predictions, axis=0)[0, 0]
        
        # Calculate Confidence Score
        confidence_map = mean_pred * (1 - variance)
        global_score = float(np.mean(confidence_map))
        
        # Generate Heatmap with overlay
        heatmap_path = generate_heatmap(mean_pred, variance, image)
        
        return {
            "score": global_score,
            "heatmapUrl": heatmap_path,
            "stats": {
                "mean_variance": float(np.mean(variance)),
                "max_confidence": float(np.max(confidence_map)),
            }
        }

    def predict_simple(self, image_file):
        """
        Single forward pass inference (faster, no uncertainty).
        
        Pipeline:
        1. Input Image → Preprocess (resize, normalize)
        2. Forward Pass → Segmentation Head (U-Net) → Raw Logits
        3. Sigmoid → Pixel-wise Probabilities (0-1)
        4. Generate Heatmap Visualization
        
        Returns:
            - heatmapUrl: path to saved heatmap
            - score: mean probability (confidence)
            - probability_map: raw numpy array of probabilities
        """
        # 1. Preprocess input image
        input_tensor = preprocess_image(image_file)  # Shape: (1, 3, 256, 256)
        
        # 2. Forward pass through segmentation head
        self.model.eval()  # Disable dropout for deterministic output
        with torch.no_grad():
            logits = self.model(input_tensor)  # Shape: (1, 1, 256, 256)
            
            # 3. Apply sigmoid for pixel-wise probabilities
            probabilities = torch.sigmoid(logits)  # Range: 0.0 - 1.0
        
        # Convert to numpy for visualization
        prob_map = probabilities.cpu().numpy()[0, 0]  # Shape: (256, 256)
        
        # 4. Generate heatmap from probability map
        heatmap_path = generate_heatmap(prob_map)
        
        # Calculate global confidence score (mean of all probabilities)
        global_score = float(np.mean(prob_map))
        
        return {
            "score": global_score,
            "heatmapUrl": heatmap_path,
            "probability_map": prob_map  # Raw array if needed for further processing
        }

