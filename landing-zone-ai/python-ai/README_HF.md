---
title: UAV Landing Zone Detection
emoji: 🚁
colorFrom: green
colorTo: red
sdk: gradio
sdk_version: 4.0.0
app_file: app_gradio.py
pinned: false
license: mit
---

# UAV Safe Landing Zone Detection

AI-powered detection of safe landing zones for UAVs/drones using semantic segmentation with uncertainty estimation.

## Features
- ResNet18-UNet architecture
- Monte Carlo Dropout for uncertainty estimation
- Real-time confidence heatmap overlay
- Green = Safe, Red = Unsafe

## Usage
Upload a UAV aerial image to detect and visualize safe landing zones.
