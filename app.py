import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the updated CNN architecture
from advanced_pruning import PrunableLinear, PrunableConv2d, PrunableNet 

# --- Configuration ---
st.set_page_config(page_title="Self-Pruning AI", page_icon="🧠", layout="wide")

st.title("🧠 Self-Pruning AI: Edge Deployment Profiler")
st.write("Upload an image to test the AI, or go to Tab 2 to look inside the AI's visual cortex!")

# --- Session State for History ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- UI Layout: SIDEBAR ---
st.sidebar.header("🧠 Select AI Brain Level")
model_choice = st.sidebar.selectbox(
    "Choose Pruning Level:", 
    [
        "Aggressive Pruning (~91% Saved)", 
        "Medium Pruning (~68% Saved)", 
        "Dense Baseline (0% Saved)"
    ]
)

st.sidebar.divider()

with st.sidebar:
    st.header("🕰️ Inference History")
    if len(st.session_state.history) == 0:
        st.caption("No images analyzed yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.image(item["image"], width=100)
            st.write(f"**{item['prediction']}** ({item['confidence']:.1f}%)")
            st.divider()

# --- Load Model ---
@st.cache_resource
def load_model(choice):
    device = torch.device("cpu") 
    model = PrunableNet()
    try:
        if choice == "Dense Baseline (0% Saved)":
            model.load_state_dict(torch.load('model_baseline.pth', map_location=device, weights_only=True))
        elif choice == "Medium Pruning (~68% Saved)":
            model.load_state_dict(torch.load('model_medium.pth', map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load('model_aggressive.pth', map_location=device, weights_only=True))
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading {choice} model. Run 'python advanced_pruning.py' to generate all 3 files!")
        return None

model = load_model(model_choice)

CLASSES = ('Airplane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==========================================
# FILE UPLOADER 
# ==========================================
st.write("---")
st.write("**📸 Step 1: Provide an image to analyze**")
col_up, col_cam = st.columns(2)
with col_up:
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
with col_cam:
    camera_file = st.camera_input("...or use your webcam!")

active_file = uploaded_file or camera_file
image = None
if active_file is not None:
    image = Image.open(active_file).convert('RGB')

st.write("---")

# === THE TABS ===
tab1, tab2 = st.tabs(["🎯 Live Inference & AI X-Ray", "👁️ Inside the AI's Visual Cortex"])

# ==========================================
# TAB 1: LIVE INFERENCE & EYE-TRACKER
# ==========================================
with tab1:
    if image is None:
        st.info("👆 Please upload an image or take a photo above to see the AI X-Ray in action.")
    elif model is not None:
        col1, col2 = st.columns([1, 1.5]) 
        
        with col1:
            st.image(image, caption="Input Image", use_container_width=True)
            xray_mode = st.toggle("🔍 Enable AI Eye-Tracker (X-Ray Mode)", help="Calculates pixel gradients to show exactly what the AI is looking at.")
            
        with col2:
            st.write("### Analysis Results")
            tensor = transform(image).unsqueeze(0)
            
            start_time = time.time()
            if xray_mode:
                tensor.requires_grad_()
                outputs = model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                target_class = outputs[0].argmax()
                outputs[0, target_class].backward()
                
                saliency, _ = torch.max(tensor.grad.data.abs(), dim=1)
                saliency = saliency.squeeze().cpu().numpy()
                saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
            else:
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    
            inference_time = (time.time() - start_time) * 1000 
            
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()
            top_prob, top_catid = torch.topk(probabilities, 3)
            top3_classes = [CLASSES[i] for i in top_catid.tolist()]
            top3_probs = [p * 100 for p in top_prob.tolist()]
            
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Primary Prediction", value=top3_classes[0], delta=f"{top3_probs[0]:.1f}% Conf")
            m2.metric(label="⚡ Latency", value=f"{inference_time:.2f} ms")
            m3.metric(label="🎲 Model Entropy", value=f"{entropy:.2f}")
            
            if len(st.session_state.history) == 0 or st.session_state.history[-1]["name"] != getattr(active_file, 'name', 'webcam_capture'):
                st.session_state.history.append({
                    "name": getattr(active_file, 'name', 'webcam_capture'),
                    "image": image,
                    "prediction": top3_classes[0],
                    "confidence": top3_probs[0]
                })
            
            if xray_mode:
                st.write("---")
                st.write("##### 🔬 X-Ray Vision: What the AI sees")
                st.caption("Bright yellow spots are the exact pixels that caused the AI to make its prediction.")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(image.resize((32, 32)), alpha=0.4) 
                ax.imshow(saliency, cmap='hot', alpha=0.6) 
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.write("---")
                chart_data = pd.DataFrame({"Confidence (%)": top3_probs, "Class": top3_classes}).set_index("Class")
                st.bar_chart(chart_data, color="#2ca02c", height=200)

# ==========================================
# TAB 2: THE VISUAL CORTEX INSPECTOR
# ==========================================
with tab2:
    st.header("👁️ The Visual Cortex Inspector")
    st.write("A Convolutional Neural Network (CNN) sees the world through multiple 'Feature Filters' (detecting edges, colors, shapes). When we prune the AI, we are physically deleting these filters to save memory.")
    
    if image is None:
        st.info("👆 Please upload an image to see how the AI's filters process it.")
    elif model is not None:
        st.write("Slide the threshold below to watch the AI's visual filters get deleted in real-time!")
        
        # Interactive Pruning Slider
        user_threshold = st.slider("🧹 Deletion Threshold (Lambda)", min_value=0.001, max_value=0.100, value=0.010, step=0.001, format="%.3f")
        
        # Find the first PrunableConv2d layer dynamically
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, PrunableConv2d):
                first_conv = module
                break
                
        if first_conv is not None:
            # Grab the custom gate scores from your architecture
            gates = torch.sigmoid(first_conv.gate_scores).detach().cpu().numpy().flatten()
            total_filters = len(gates)
            active_filters = np.sum(gates >= user_threshold)
            
            c1, c2 = st.columns(2)
            c1.metric("Active Visual Filters", f"{active_filters} / {total_filters}")
            c2.metric("Layer Sparsity", f"{(1 - (active_filters/total_filters))*100:.1f}% Deleted")
            
            # Run the image through just the first layer to see the raw feature maps
            tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                try:
                    feature_maps = first_conv(tensor)[0] # Shape: [Filters, Height, Width]
                    
                    st.write("---")
                    st.subheader("Raw Feature Maps")
                    
                    # Create a dynamic grid based on the number of filters
                    cols = 4
                    rows = int(np.ceil(total_filters / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(8, rows * 2))
                    axes = axes.flatten()
                    
                    for i in range(total_filters):
                        ax = axes[i]
                        # If the gate score is below the threshold, the filter is "pruned"
                        if gates[i] < user_threshold:
                            # Show a blank square for deleted filters
                            blank = np.zeros((feature_maps.shape[1], feature_maps.shape[2]))
                            ax.imshow(blank, cmap='gray')
                            ax.set_title("DELETED", color='red', fontsize=10, fontweight='bold')
                        else:
                            # Show the actual feature map processing the image
                            fm = feature_maps[i].cpu().numpy()
                            ax.imshow(fm, cmap='viridis')
                            ax.set_title(f"Filter {i}", color='green', fontsize=10)
                        
                        ax.axis('off')
                        
                    # Hide any extra empty subplots
                    for j in range(total_filters, len(axes)):
                        axes[j].axis('off')
                        
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error("Could not run the partial forward pass. Ensure your PrunableConv2d accepts a standard tensor input.")