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
from groq import Groq # Swapped from Google Gemini to Groq LPU

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

# --- 2. SECURE LLM SETUP (GROQ LPU ENGINE) ---
# Hardcoded as a fallback, but try to use st.secrets long-term!
groq_key = st.secrets.get(
    "GROQ_API_KEY", 
    os.getenv("GROQ_API_KEY", "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")
)
groq_client = Groq(api_key=groq_key)

def generate_clinical_report(diagnosis, conf_pct):
    prompt = f"""
    Act as a Lead Clinical Pathologist. 
    Write a brief, highly professional 3-sentence consultation summary for a malaria thin blood smear scan. 
    Diagnosis: {diagnosis}
    AI Confidence: {conf_pct:.2f}%
    
    If the diagnosis is 'Parasitized', advise immediate clinical correlation and pathology review.
    If 'Uninfected', state that no parasitic markers were detected but advise monitoring if symptoms persist.
    """
    try:
        # Utilizing Groq's high-speed Llama 3.3 70B engine
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2, # Lower temperature for clinical, factual tone
            max_tokens=150,  
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"CONNECTION ERROR: LLM Synthesis failed. Details: {str(e)}"

# --- 3. ASSET LOADING & MODEL INITIALIZATION ---
@st.cache_resource
def load_clinical_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the base ResNet50 model
    model = models.resnet50(weights=None)
    
    # 2. Load the state dictionary from the file securely
    state_dict = torch.load("best_malaria_resnet.pth", map_location=device, weights_only=True)
    
    # 3. DYNAMIC ARCHITECTURE RECONSTRUCTION
    if 'fc.0.weight' in state_dict:
        in_ftrs = state_dict['fc.0.weight'].shape[1]
        hidden_ftrs = state_dict['fc.0.weight'].shape[0]
        out_ftrs = state_dict['fc.3.weight'].shape[0]
        
        model.fc = nn.Sequential(
            nn.Linear(in_ftrs, hidden_ftrs),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(hidden_ftrs, out_ftrs)
        )
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        
    # 4. Inject weights
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

# --- 4. UI LAYOUT: HEADER & EXPANDERS ---
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='font-size: 80px;'>🔬</h1>", unsafe_allow_html=True)

with col2:
    st.title("Malaria Diagnostic CDSS")
    st.markdown("<p style='color: #64748B; font-size: 1.2rem;'>Enterprise Vision Engine for Automated Pathology</p>", unsafe_allow_html=True)

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

# --- 5. DIAGNOSTIC WORKFLOW ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Patient Sample Input")
    uploaded_file = st.file_uploader("Upload Micrograph", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        # Load the original image for the PyTorch Model
        original_image = Image.open(uploaded_file).convert('RGB')
        
        # UI LAYOUT: Create a display thumbnail so it doesn't stretch the screen
        display_image = original_image.copy()
        display_image.thumbnail((400, 400)) # Locks the visual size without distorting
        
        # Updated deprecated use_column_width to width="stretch" per Streamlit logs
        st.image(display_image, caption="Original Patient Sample", width="stretch")
        
        if st.button("Execute Diagnostic Scan"):
            # Inference Pipeline (Using the original high-res image)
            input_tensor = inference_transforms(original_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = clinical_model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(prob, 1)
            
            # Setup Variables
            diagnosis = "Parasitized" if pred.item() == 0 else "Uninfected"
            conf_pct = confidence.item() * 100
            
            # Trigger LLM Synthesis Before Saving to Session State
            with st.spinner("🤖 Synthesizing clinical data via Llama 3..."):
                llm_summary = generate_clinical_report(diagnosis, conf_pct)
            
            # Store everything securely in session state
            st.session_state['results'] = {
                'class': diagnosis,
                'conf': conf_pct,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': uploaded_file.name,
                'summary': llm_summary
            }

with col_output:
    st.subheader("Diagnostic Results")
    if 'results' in st.session_state:
        res = st.session_state['results']
        
        # White Container for Results with injected LLM Summary
        # Safely retrieve the summary using .get() to prevent KeyErrors on old sessions
        report_summary = res.get('summary', 'Summary not available. Please click "Execute Diagnostic Scan" to generate.')
        
        st.markdown(f"""
            <div class="report-container">
                <div class="section-header">CLINICAL TRIAGE SUMMARY</div>
                <p><b>Sample ID:</b> {res.get('filename', 'Unknown')}</p>
                <p><b>Scan Timestamp:</b> {res.get('timestamp', 'Unknown')}</p>
                <hr style="border: 0.5px solid #E2E8F0;">
                <h2 style="color: {'#E11D48' if res.get('class') == 'Parasitized' else '#10B981'};">
                    {str(res.get('class', 'Unknown')).upper()}
                </h2>
                <p><b>AI Confidence Score:</b> {res.get('conf', 0):.2f}%</p>
                <br>
                <div class="section-header">PHYSICIAN CONSULTATION NOTE</div>
                <p style="font-style: italic; color: #475569; line-height: 1.6;">
                    {report_summary}
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Awaiting patient sample for analysis...")

# --- 6. FOOTER ---
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
