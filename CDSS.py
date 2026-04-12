# =============================================================================
# CLINICAL DECISION SUPPORT SYSTEM (CDSS) - MALARIA DIAGNOSTICS
# Architect: Moses Mudiaga Effeyotah | INFO 6147 CAPSTONE
# =============================================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai
import datetime

# --- 1. System Configuration & Premium UI ---
st.set_page_config(page_title="Malaria Diagnostic AI", page_icon="🔬", layout="wide")

st.markdown("""
    <style>
    /* 1. Global Deep Blue Background */
    .stApp { background-color: #0F172A; color: #F1F5F9; }

    /* 2. Fix Upload Button Text Color (Make it Black) */
    /* Streamlit uses buttons for file uploads and standard actions */
    button, [data-testid="stFileUploadDropzone"] p {
        color: #000000 !important;
        font-weight: 600;
    }

    /* 3. Global Text Visibility */
    .stMarkdown, p, span, label { color: #F1F5F9 !important; }

    /* 4. High-Authority Title */
    .stTitle { 
        border-bottom: 3px solid #E02035; 
        padding-bottom: 10px; 
        color: #FFFFFF !important; 
        font-weight: 900;
    }

    /* 5. Metrics: Emerald Green for Primary Data */
    [data-testid="stMetricValue"] { color: #10B981 !important; font-size: 2.5rem !important; }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; }

    /* 6. The Pathology Report (The "Parchment") */
    /* We override text here back to dark blue for readability on white paper */
    .report-box {
        background-color: #FFFFFF; 
        color: #0F172A !important; 
        padding: 30px;
        border-radius: 4px;
        border-left: 10px solid #E02035;
    }
    .report-box div, .report-box span, .report-box p { 
        color: #0F172A !important; 
    }
    
    .report-footer {
        color: #64748B !important;
        font-size: 0.8rem;
    }
</style>
    """, unsafe_allow_html=True)

# --- 2. SECURE API Initialization ---
# REPLACEMENT: Fetching from Secrets (Required for GitHub safety)
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("API Key Missing: Please add 'GEMINI_API_KEY' to Streamlit Secrets.")

# --- 3. Vision Model Architecture ---
@st.cache_resource
def load_medical_model():
    device = torch.device("cpu") # Optimized for Streamlit Cloud CPUs
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("best_malaria_resnet.pth", map_location=device))
    model.eval()
    return model, device

vision_model, computation_device = load_medical_model()

# --- 4. Clinical Report Synthesis ---
def generate_clinical_report(diagnosis, conf_pct):
    # Fixed the variable naming and f-string structure
    prompt = (
        f"ROLE: Expert Clinical Hematologist. "
        f"CONTEXT: Peripheral blood smear analysis. "
        f"FINDINGS: {diagnosis} ({conf_pct:.2f}% Confidence). "
        f"TASK: Provide a professional pathology report with sections: "
        f"Clinical Summary, Morphological Findings, and Recommended Protocols. "
        f"TONE: Authoritative, medical precision. Plain text only."
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text if response else "Synthesis failed."

# --- 5. Main Logic ---
st.title("🔬 AUTOMATED MALARIA DIAGNOSTICS")
st.markdown("**CDSS PIPELINE** | *ResNet-50 Vision + Gemini Synthesis*")

uploaded_file = st.file_uploader("UPLOAD SPECIMEN", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.2], gap="large")
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.markdown('<div class="section-header">I. PATIENT SPECIMEN</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">II. DIAGNOSTIC INFERENCE</div>', unsafe_allow_html=True)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = vision_model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_class = torch.max(probs, 0)

        diagnosis = "Parasitized" if predicted_class.item() == 0 else "Uninfected"
        conf_pct = confidence.item() * 100

        m1, m2 = st.columns(2)
        m1.metric("FINDING", diagnosis.upper())
        m2.metric("CONFIDENCE", f"{conf_pct:.2f}%")

        if diagnosis == "Parasitized":
            st.error("🚨 CRITICAL ALERT: Immediate clinical correlation required.")
        else:
            st.success("⚕️ BENIGN FINDING: Morphology clear.")

    st.divider()
    st.markdown('<div class="section-header">III. PHYSICIAN CONSULTATION SUMMARY</div>', unsafe_allow_html=True)
    
    with st.spinner("🤖 Synthesizing report..."):
        try:
            report_text = generate_clinical_report(diagnosis, conf_pct)
        except Exception as e:
            report_text = f"CONNECTION ERROR: API link severed. Details: {str(e)}"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S EST")
        st.markdown(f"""
            <div class="report-box">
                <div style="border-bottom: 2px solid #E2E8F0; margin-bottom: 20px; font-weight: bold;">
                    OFFICIAL PATHOLOGY REPORT | {timestamp}
                </div>
                {report_text}
                <div style="font-size: 0.8em; border-top: 1px dashed #CCC; margin-top: 20px; text-align: right;">
                    DIGITALLY SIGNED: CDSS-ALPHA VISION
                </div>
            </div>
        """, unsafe_allow_html=True)
