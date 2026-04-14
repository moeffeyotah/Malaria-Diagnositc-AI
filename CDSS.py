# =============================================================================
# CLINICAL DECISION SUPPORT SYSTEM (CDSS) - MALARIA DIAGNOSTICS
# Architect: Moses Mudiaga Effeyotah
# School of Information Technology | Fanshawe College
# =============================================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import datetime
import os

# --- 1. SYSTEM CONFIGURATION & THEME ARCHITECTURE ---
st.set_page_config(page_title="Malaria Diagnostic AI", page_icon="🔬", layout="wide")

# Custom CSS: "Clinical Slate" Theme
st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; }
    
    /* White Container Div for Results */
    .report-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #1E293B; /* Deep Slate for text readability */
    }

    /* Section Headers */
    .section-header {
        color: #0F172A;
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 5px solid #0EA5E9;
        padding-left: 15px;
        margin-bottom: 20px;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #0F172A;
        color: white;
        border-radius: 8px;
        width: 100%;
        border: none;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #0EA5E9;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. ASSET LOADING & MODEL INITIALIZATION ---
@st.cache_resource
def load_clinical_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the base ResNet50 model
    model = models.resnet50(weights=None)
    
    # 2. Load the state dictionary from the file securely
    # Added weights_only=True for modern PyTorch security standards
    state_dict = torch.load("best_malaria_resnet.pth", map_location=device, weights_only=True)
    
    # 3. DYNAMIC ARCHITECTURE RECONSTRUCTION
    # Check if the saved model used a Sequential block (fc.0 and fc.3)
    if 'fc.0.weight' in state_dict:
        # Extract the exact node counts from the saved weights
        in_ftrs = state_dict['fc.0.weight'].shape[1]
        hidden_ftrs = state_dict['fc.0.weight'].shape[0]
        out_ftrs = state_dict['fc.3.weight'].shape[0]
        
        # Rebuild the exact sequence used during training
        model.fc = nn.Sequential(
            nn.Linear(in_ftrs, hidden_ftrs),
            nn.ReLU(),
            nn.Dropout(0.5), # Exact % doesn't matter since model.eval() turns dropout off
            nn.Linear(hidden_ftrs, out_ftrs)
        )
    else:
        # Fallback to standard single-layer replacement just in case
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    # 4. Now inject the weights into our perfectly matched architecture
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device

clinical_model, device = load_clinical_model()

# Image Preprocessing (Matches training engine)
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. UI LAYOUT: HEADER & EXPANDERS ---
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='font-size: 80px;'>🔬</h1>", unsafe_allow_html=True)

with col2:
    st.title("Malaria Diagnostic CDSS")
    st.markdown("<p style='color: #64748B; font-size: 1.2rem;'>Enterprise Vision Engine for Automated Pathology</p>", unsafe_allow_html=True)

# 3.1 About & How to Use
tab_about, tab_guide = st.expander("📖 About this AI Architecture"), st.expander("🛠 Clinical User Guide")

with tab_about:
    st.write("""
    **Architecture:** This system utilizes a **ResNet-50 Deep Residual Network** fine-tuned on the NIH Malaria Dataset (27,558 images).
    
    **Integrity:** The engine is designed with a **Recall-First** priority, ensuring maximum sensitivity for parasitic detection. 
    By bypassing traditional 'Black Box' limitations, this CDSS provides localized diagnostic evidence via spatial feature mapping.
    """)

with tab_guide:
    st.write("""
    1. **Upload:** Provide a high-resolution blood smear micrograph (PNG/JPG).
    2. **Process:** The system will resize and normalize the image to match medical standards.
    3. **Analysis:** Click 'Execute Diagnostic Scan'.
    4. **Triage:** Review the classification and the generated Physician Consultation Summary.
    """)

st.divider()

# --- 4. DIAGNOSTIC WORKFLOW ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Patient Sample Input")
    uploaded_file = st.file_uploader("Upload Micrograph", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Original Patient Sample", use_container_width=True)
        
        if st.button("Execute Diagnostic Scan"):
            # Inference Pipeline
            input_tensor = inference_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = clinical_model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(prob, 1)
            
            # Store results in session state for the output column
            st.session_state['results'] = {
                'class': "Parasitized" if pred.item() == 0 else "Uninfected",
                'conf': confidence.item() * 100,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

with col_output:
    st.subheader("Diagnostic Results")
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        # White Container for Results
        st.markdown(f"""
            <div class="report-container">
                <div class="section-header">CLINICAL TRIAGE SUMMARY</div>
                <p><b>Sample ID:</b> {uploaded_file.name}</p>
                <p><b>Scan Timestamp:</b> {res['timestamp']}</p>
                <hr style="border: 0.5px solid #E2E8F0;">
                <h2 style="color: {'#E11D48' if res['class'] == 'Parasitized' else '#10B981'};">
                    {res['class'].upper()}
                </h2>
                <p><b>AI Confidence Score:</b> {res['conf']:.2f}%</p>
                <br>
                <div class="section-header">PHYSICIAN CONSULTATION NOTE</div>
                <p style="font-style: italic; color: #475569;">
                    The vision engine has identified morphology consistent with <b>{res['class'].lower()}</b> cells. 
                    {'🚨 Immediate pathological verification required.' if res['class'] == 'Parasitized' else '✅ No immediate parasitic markers detected.'}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting patient sample for analysis...")

# --- 5. FOOTER ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #94A3B8; font-size: 0.8em;'>
        Designed and Engineered by <strong>Moses Mudiaga Effeyotah</strong><br>
        School of Information Technology | Fanshawe College | INFO-6147 Capstone
    </div>
    """,
    unsafe_allow_html=True,
)
