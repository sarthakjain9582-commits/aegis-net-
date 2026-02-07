"""
Gradio Demo for UAV Landing Zone Detection
Deploy on Hugging Face Spaces - Using YOLOv8-Nano
"""
import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Import model
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.yolov8_landing import YOLOv8SegmentationWrapper

# Global model instance
MODEL = None

def load_model():
    global MODEL
    if MODEL is None:
        MODEL = YOLOv8SegmentationWrapper()
        print("YOLOv8-Nano Segmentation loaded!")
    return MODEL

def preprocess(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def generate_overlay(original, mean_pred, variance):
    """Generate green/red overlay on original image"""
    H, W = mean_pred.shape
    
    # Resize original to match
    original_resized = cv2.resize(np.array(original), (W, H))
    
    # Create heatmap: Green (safe) to Red (unsafe)
    heatmap = np.zeros((H, W, 3), dtype=np.uint8)
    heatmap[:, :, 1] = (mean_pred * 255).astype(np.uint8)  # Green = high confidence
    heatmap[:, :, 2] = ((1 - mean_pred) * 255).astype(np.uint8)  # Red = low confidence
    
    # Blend
    overlay = cv2.addWeighted(original_resized, 0.6, heatmap, 0.4, 0)
    return overlay

def predict(image, mc_samples=10):
    """
    Main prediction function for Gradio using YOLOv8-Nano.
    """
    model = load_model()
    
    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Run prediction with uncertainty
    mean_pred, variance = model.predict_with_uncertainty(image_np, n_passes=mc_samples)
    
    # Handle empty predictions
    if mean_pred.sum() == 0:
        mean_pred = np.ones_like(mean_pred) * 0.5
        variance = np.ones_like(variance) * 0.5
    
    # Confidence map and score
    confidence_map = mean_pred * (1 - variance)
    global_score = float(np.mean(confidence_map))
    mean_uncertainty = float(np.mean(variance))
    
    # Generate overlay
    overlay = generate_overlay(image, mean_pred, variance)
    
    # Format outputs
    confidence_text = f"{global_score:.2%}"
    uncertainty_text = f"{mean_uncertainty:.4f}"
    
    return overlay, confidence_text, uncertainty_text

# Gradio Interface
with gr.Blocks(title="UAV Landing Zone Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚁 UAV Safe Landing Zone Detection
    
    Upload a UAV/drone aerial image to detect safe landing zones.
    
    - **Green areas**: Safe for landing (high confidence)
    - **Red areas**: Unsafe (low confidence)
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload UAV Image")
            mc_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1, 
                                  label="MC Dropout Passes (more = better uncertainty estimate)")
            predict_btn = gr.Button("Analyze Landing Zones", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Confidence Heatmap Overlay")
            with gr.Row():
                confidence_output = gr.Textbox(label="Confidence Score")
                uncertainty_output = gr.Textbox(label="Mean Uncertainty")
    
    predict_btn.click(
        fn=predict,
        inputs=[input_image, mc_slider],
        outputs=[output_image, confidence_output, uncertainty_output]
    )
    
    gr.Markdown("""
    ---
    ### How it works:
    1. **YOLOv8-Nano** segmentation model analyzes the image
    2. Multiple augmented passes estimate uncertainty
    3. **Confidence = Mean × (1 - Variance)** per pixel
    4. Results are overlaid on the original image (Green = Safe, Red = Unsafe)
    """)

if __name__ == "__main__":
    demo.launch()
