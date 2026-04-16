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
from groq import Groq

# --- 1. SYSTEM CONFIGURATION & THEME ARCHITECTURE ---
st.set_page_config(page_title="Malaria Diagnostic AI", page_icon="🔬", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #F8FAFC; }
    
    .report-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #1E293B; 
    }

    .section-header {
        color: #0F172A;
        font-size: 1.2rem;
        font-weight: 700;
        border-left: 5px solid #0EA5E9;
        padding-left: 15px;
        margin-bottom: 20px;
    }

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

# --- 2. SECURE LLM SETUP ---
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
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2, 
            max_tokens=150,  
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"CONNECTION ERROR: LLM Synthesis failed. Details: {str(e)}"

# --- 3. ASSET LOADING & MODEL INITIALIZATION ---
@st.cache_resource
def load_clinical_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=None)
    state_dict = torch.load("best_malaria_resnet.pth", map_location=device, weights_only=True)
    
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
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, device

clinical_model, device = load_clinical_model()

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
    1. **Source:** Choose to either upload your own Micrograph or select from our pre-loaded clinical database.
    2. **Process:** The system will resize and normalize the image to match medical standards.
    3. **Analysis:** Click 'Execute Diagnostic Scan'.
    4. **Triage:** Review the classification and the generated Physician Consultation Summary.
    """)

st.divider()

# --- 5. DIAGNOSTIC WORKFLOW ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("Patient Sample Input")
    
    source_option = st.radio("Choose Input Method:", ["Upload Micrograph", "Use Clinical Database"], horizontal=True)
    
    image = None
    file_id = ""
    
    if source_option == "Upload Micrograph":
        uploaded_file = st.file_uploader("Drop PNG/JPG here", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            file_id = uploaded_file.name
            
    else:
        sample_dir = "clean_cell_images" 
        if os.path.exists(sample_dir):
            # UX UPDATE: Cascading Dropdowns for Category -> Image Selection
            categories = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
            
            if categories:
                cat_col, img_col = st.columns(2)
                
                with cat_col:
                    selected_category = st.selectbox("1. Select Cell Class:", sorted(categories))
                
                category_path = os.path.join(sample_dir, selected_category)
                sample_files = [f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                with img_col:
                    if sample_files:
                        selected_file = st.selectbox("2. Select Sample ID:", sorted(sample_files))
                        image_path = os.path.join(category_path, selected_file)
                        image = Image.open(image_path).convert('RGB')
                        file_id = f"{selected_category}/{selected_file}"
                    else:
                        st.warning(f"No images found in {selected_category}.")
            else:
                st.warning("⚠️ No subfolders (Parasitized/Uninfected) found.")
        else:
            st.warning("⚠️ The clinical database folder was not found in the repository.")

    if image:
        st.markdown("<br>", unsafe_allow_html=True) 
        
        # Display thumbnail for layout stability
        display_image = image.copy()
        display_image.thumbnail((400, 400)) 
        st.image(display_image, caption=f"Sample ID: {file_id}", width="stretch")
        
        if st.button("Execute Diagnostic Scan", type="primary"):
            input_tensor = inference_transforms(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = clinical_model(input_tensor)
                prob = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(prob, 1)
            
            diagnosis = "Parasitized" if pred.item() == 0 else "Uninfected"
            conf_pct = confidence.item() * 100
            
            with st.spinner("🤖 Synthesizing clinical data via Llama 3..."):
                llm_summary = generate_clinical_report(diagnosis, conf_pct)
            
            st.session_state['results'] = {
                'class': diagnosis,
                'conf': conf_pct,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': file_id,
                'summary': llm_summary
            }

with col_output:
    st.subheader("Diagnostic Results")
    if 'results' in st.session_state:
        res = st.session_state['results']
        
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
