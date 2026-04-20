# 🧠 Self-Pruning Convolutional Neural Network (CNN)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**Live Demo:** `https://gzynokkzk3qxjhndurn2ug.streamlit.app/`

## 📌 Overview
Deep learning models are often too large and memory-intensive to deploy on edge devices (mobile, IoT). Standard models waste compute power on redundant neural connections. 

This project solves that by implementing a **Custom Self-Pruning CNN** using an $L_1$-regularized gating mechanism. During training, the network organically learns to "shut off" useless parameters, dynamically compressing its own memory footprint while maintaining high accuracy.

## 🚀 Key Features
* **Live Inference & AI Eye-Tracker:** An interactive dashboard that accepts image uploads/webcam capture and generates **Saliency Maps (X-Ray Vision)** using backpropagation to show exactly which pixels the AI is looking at.
* **The INT8 Quantization Studio:** A visual simulator that crushes the mathematical precision of the model (from 32-bit floats to 8-bit integers) and visually demonstrates the resulting loss of visual fidelity.
* **Edge Hardware Profiler:** Dynamically translates pruned weights into estimated real-world Edge CCTV Video FPS, Thermal outputs, and Battery life savings.
* **Model Entropy Tracking:** Calculates prediction entropy to detect Out-of-Distribution (OOD) uncertainty.

## 🧠 Architectural Highlights
### 1. The $L_1$ Gating Mechanism
Custom `PrunableLinear` and `PrunableConv2d` layers were built from scratch in PyTorch. Each layer includes a learnable "Gate Score" parameterized by a sigmoid function. An $L_1$ penalty is applied to these gates during training, forcing the network to optimize for both accuracy and extreme sparsity.

### 2. The "Greedy" Feature Extractor
The model organically learned a "Bottleneck" architecture. It protected 100% of its Convolutional spatial filters to maintain high-resolution 'vision', while ruthlessly pruning **99.5%+** of its dense classifier nodes where the most memory overhead exists.

### 3. Bias Correction via Augmentation
Initial testing revealed a spurious correlation (the model associated "Blue Sky" backgrounds with the "Bird" class). This was corrected by implementing an aggressive data augmentation pipeline (Random Rotation, Crop, Horizontal Flip, and Color Jitter) to destroy background bias and force the network to learn actual spatial features.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Frontend / UI:** Streamlit, Pandas, Matplotlib, PIL
* **Deployment:** Docker, Streamlit Community Cloud

## 💻 Local Installation & Usage

### Option 1: Docker (Recommended)
The easiest way to run the application is via the included Docker container.

```bash
# 1. Clone the repository
git clone [https://github.com/Purvi1411/Self-Pruning-Convolutional-Neural-Network-CNN-.git](https://github.com/Purvi1411/Self-Pruning-Convolutional-Neural-Network-CNN-.git)
cd Self-Pruning-Convolutional-Neural-Network-CNN-

# 2. Build the Docker image
docker build -t ai-pruning-project .

# 3. Run the container
docker run -p 8501:8501 -it ai-pruning-project