# =============================================================================
# CLINICAL DECISION SUPPORT SYSTEM (CDSS) - MALARIA DIAGNOSTICS
# Architect: Moses Mudiaga Effeyotah
# =============================================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai

# =============================================================================
# 1. System Configuration & Premium UI Architecture
# =============================================================================
st.set_page_config(page_title="Malaria Diagnostic AI", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    /* Base Application Theme: Clean Clinical Slate */
    .stApp { background-color: #F8FAFC; color: #0F172A; }
    
    /* Hide Streamlit Default Branding for SaaS feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Typography: Crisp, modern sans-serif for UI, tight letter spacing */
    h1, h2, h3, h4, h5 { 
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
        color: #0F172A; 
        font-weight: 800; 
        letter-spacing: -0.5px; 
    }
    
    /* Master Class Header Styling */
    .stTitle { 
        border-bottom: 4px solid #E02035; 
        padding-bottom: 15px; 
        margin-bottom: 30px;
        text-transform: uppercase;
        font-weight: 900;
    }

    /* Section Subheaders */
    .section-header {
        color: #334155;
        font-size: 1.25rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        border-left: 4px solid #94A3B8;
        padding-left: 10px;
    }

    /* Official Pathology Report Container */
    .report-box {
    background-color: #FFFFFF;
    border-left: 10px solid #0F172A; /* Thick medical navy bar */
    padding: 30px;
    border-radius: 4px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    font-family: 'Courier New', Courier, monospace; /* Typewriter feel */
}
.report-header {
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 20px;
    padding-bottom: 10px;
    font-weight: bold;
}

    /* Report Metadata Header */
    .report-header {
        border-bottom: 2px solid #E2E8F0;
        margin-bottom: 25px;
        padding-bottom: 15px;
        font-weight: 800;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1rem;
        display: flex;
        justify-content: space-between;
    }

    /* Digital Signature Styling */
    .report-footer {
        font-size: 0.8em; 
        border-top: 1px dashed #CBD5E1; 
        padding-top: 15px;
        margin-top: 30px;
        color: #94A3B8;
        font-weight: bold;
        letter-spacing: 1px;
        text-align: right;
    }

    /* Standardize Image Container */
    [data-testid="stImage"] {
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }
    
    /* Enhance Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize API Client
GENAI_API_KEY = "AIzaSyAFXzGD3tqdY9fFmgdBQmz_dvcELq6-nTY"
client = genai.Client(api_key=GENAI_API_KEY)


# =============================================================================
# 2. Vision Model Loader
# =============================================================================
@st.cache_resource
def load_medical_model():
    """Loads the fine-tuned ResNet-50 model into memory."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2)
    )

    model.load_state_dict(
        torch.load("best_malaria_resnet.pth", map_location=device, weights_only=True)
    )
    model = model.to(device)
    model.eval()

    return model, device


vision_model, computation_device = load_medical_model()


# =============================================================================
# 3. Data Transformations
# =============================================================================
medical_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# =============================================================================
# 4. LLM Reporting Engine
# =============================================================================
def generate_clinical_report(diagnosis, confidence):
    """Generates a physician-level summary using Gemini 2.5 Flash."""
    prompt = f"""
    ROLE: Expert Clinical Hematologist.
    CONTEXT: Computer Vision analysis of a peripheral blood smear.
    FINDINGS: {diagnosis} ({confidence:.2f}% System Confidence).

    TASK: Write a professional, concise clinical summary for the attending physician.
    Structure the response clearly with the following bolded sections:
    - **Clinical Summary:** - **Morphological Findings:** (Use terms like intracellular, erythrocytes, etc)
    - **Clinical Implications:**
    - **Recommended Protocols:** (e.g., CBC, manual microscopic review)
    
    TONE: Objective, doctor-to-doctor, clinical precision. 
    NO conversational filler. Use direct, authoritative grammar. Formatted in plain text or simple markdown.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    if response and response.text:
        return response.text
    else:
        return "ERROR: API connectivity established, but content synthesis was interrupted."


# =============================================================================
# 5. Front-End Interface
# =============================================================================
st.title("🔬 AUTOMATED MALARIA DIAGNOSTICS")
st.markdown("**ARCHITECT: MOSES MUDIAGA EFFEYOTAH** | *INFO 6147 CAPSTONE*")

st.divider()

uploaded_file = st.file_uploader(
    "UPLOAD BLOOD SMEAR SPECIMEN (JPG/PNG)", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2], gap="large")

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.markdown(
            '<div class="section-header">I. PATIENT SPECIMEN</div>',
            unsafe_allow_html=True,
        )
        st.image(image, use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-header">II. DIAGNOSTIC INFERENCE</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Analyzing morphology via ResNet-50..."):

            # Forward pass through ResNet-50
            input_tensor = medical_transform(image).unsqueeze(0).to(computation_device)
            with torch.no_grad():
                outputs = vision_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_class = torch.max(probabilities, 0)

            diagnosis = ["Parasitized", "Uninfected"][predicted_class.item()]
            conf_pct = confidence.item() * 100

            # Render High-Status Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="Primary Finding", value=diagnosis.upper())
            with m2:
                st.metric(label="System Confidence", value=f"{conf_pct:.2f}%")

            st.write("")  # Spacer

            # Render Status Alert
            if diagnosis == "Parasitized":
                st.error(
                    "⚠️ **CRITICAL ALERT:** Immediate clinical correlation required.",
                    icon="🚨",
                )
            else:
                st.success(
                    "✅ **BENIGN FINDING:** Morphology consistent with uninfected sample.",
                    icon="⚕️",
                )

    st.divider()

    # Trigger LLM Synthesis
    st.markdown(
        '<div class="section-header">III. PHYSICIAN CONSULTATION SUMMARY</div>',
        unsafe_allow_html=True,
    )
    with st.spinner("🤖 Synthesizing clinical data via LLM..."):
        try:
            report_text = generate_clinical_report(diagnosis, conf_pct)
        except Exception as e:
            report_text = f"CONNECTION ERROR: API link severed. Details: {str(e)}"

        # Import datetime to timestamp the report
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S EST")

        # Render custom report HTML container
        st.markdown(
            f"""
            <div class="report-box">
                <div class="report-header">
                    <span>OFFICIAL PATHOLOGY REPORT - MALARIA SCREENING</span>
                    <span style="color: #94A3B8; font-size: 0.8rem;">{timestamp}</span>
                </div>
                {report_text}
                <div class="report-footer">
                    DIGITALLY SIGNED: CDSS-ALPHA VISION SYSTEM | ENCRYPTED CONNECTION
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
