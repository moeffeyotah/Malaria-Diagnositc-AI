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
import pandas as pd
import numpy as np
import cv2
import tempfile
from fpdf import FPDF
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
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0EA5E9;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. SECURE LLM SETUP (Groq LPU) ---
groq_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", "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"))
groq_client = Groq(api_key=groq_key)

def generate_clinical_report(diagnosis, conf_pct, age, travel, symptoms, triage_status):
    prompt = f"""
    Act as a Lead Clinical Pathologist. Write a brief, highly professional 3-sentence consultation summary for a malaria thin blood smear scan. 
    
    Patient Context: Age {age}, Travel History: {travel}, Symptoms: {symptoms}.
    AI Visual Diagnosis: {diagnosis}
    AI Confidence: {conf_pct:.2f}%
    Triage Status: {triage_status}
    
    Synthesize the patient context with the visual diagnosis. 
    If 'Parasitized' or 'Indeterminate', advise immediate clinical correlation. 
    If 'Uninfected', state no parasitic markers were detected but advise monitoring based on symptoms.
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.2, 
            max_tokens=200,  
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"CONNECTION ERROR: LLM Synthesis failed. Details: {str(e)}"

# --- 3. ASSET LOADING & MODEL INITIALIZATION ---
# Dictionary to store layer activations for the XAI Heatmap
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

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
    
    # Register hook on the last convolutional layer for the XAI Heatmap
    model.layer4.register_forward_hook(get_activation('layer4'))
    
    return model, device

clinical_model, device = load_clinical_model()

inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- FEATURE 1 & 3: XAI HEATMAP & PDF GENERATION ---
def generate_heatmap(original_image, act_map):
    # Process activation map into a heatmap
    heatmap = torch.mean(act_map.squeeze(), dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0) # ReLU
    heatmap /= np.max(heatmap) # Normalize
    
    # Resize to match original image
    original_cv = np.array(original_image)
    heatmap_resized = cv2.resize(heatmap, (original_cv.shape[1], original_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlay = cv2.addWeighted(cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

def create_pdf(res):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Clinical Triage Report: Malaria CDSS", ln=True, align='C')
    pdf.line(10, 20, 200, 20)
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Sample ID: {res['filename']}", ln=True)
    pdf.cell(200, 10, txt=f"Scan Timestamp: {res['timestamp']}", ln=True)
    pdf.cell(200, 10, txt=f"Patient Context: Age {res['age']} | Travel: {res['travel']}", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Diagnostic Classification: {res['class'].upper()}", ln=True)
    pdf.cell(200, 10, txt=f"Triage Status: {res['triage'].upper()}", ln=True)
    pdf.cell(200, 10, txt=f"AI Confidence Score: {res['conf']:.2f}%", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 10, txt=f"Physician Note: {res['summary']}")
    
    return pdf.output(dest='S').encode('latin1')

# --- 4. UI LAYOUT ---
st.sidebar.markdown("### 📋 Patient Context (RAG)")
patient_age = st.sidebar.number_input("Patient Age", min_value=1, max_value=120, value=35)
patient_travel = st.sidebar.text_input("Recent Travel History", "e.g., Sub-Saharan Africa")
patient_symptoms = st.sidebar.text_area("Reported Symptoms", "e.g., Intermittent fever, chills")

st.markdown("<h1 style='text-align: center;'>🔬 Malaria Diagnostic CDSS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Enterprise Vision Engine featuring Explainable AI & Batch Processing</p>", unsafe_allow_html=True)
st.divider()

# --- FEATURE 5: TABS FOR SINGLE VS BATCH PROCESSING ---
tab_single, tab_batch = st.tabs(["🩺 Single Patient Triage", "📂 High-Throughput Batch Processing"])

with tab_single:
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
                            image = Image.open(os.path.join(category_path, selected_file)).convert('RGB')
                            file_id = f"{selected_category}/{selected_file}"
            else:
                st.warning("⚠️ The clinical database folder was not found.")

        if image:
            display_image = image.copy()
            display_image.thumbnail((350, 350)) 
            st.image(display_image, caption=f"Sample ID: {file_id}", width="stretch")
            
            if st.button("Execute Diagnostic Scan", type="primary"):
                input_tensor = inference_transforms(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = clinical_model(input_tensor)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    confidence, pred = torch.max(prob, 1)
                
                diagnosis = "Parasitized" if pred.item() == 0 else "Uninfected"
                conf_pct = confidence.item() * 100
                
                # FEATURE 2: HUMAN-IN-THE-LOOP TRIAGE
                triage_status = "CLEAR" if conf_pct >= 85 else "INDETERMINATE - HUMAN REVIEW REQUIRED"
                
                # FEATURE 1: GENERATE XAI HEATMAP
                heatmap_img = generate_heatmap(image, activation['layer4'])
                
                with st.spinner("🤖 Synthesizing clinical data via Groq LPU..."):
                    llm_summary = generate_clinical_report(diagnosis, conf_pct, patient_age, patient_travel, patient_symptoms, triage_status)
                
                st.session_state['results'] = {
                    'class': diagnosis,
                    'conf': conf_pct,
                    'triage': triage_status,
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'filename': file_id,
                    'summary': llm_summary,
                    'age': patient_age,
                    'travel': patient_travel,
                    'heatmap': heatmap_img
                }

    with col_output:
        st.subheader("Diagnostic Results")
        if 'results' in st.session_state:
            res = st.session_state['results']
            status_color = "#E11D48" if res['class'] == 'Parasitized' else "#10B981"
            if "INDETERMINATE" in res['triage']: status_color = "#F59E0B" # Amber warning
            
            st.markdown(f"""
                <div class="report-container">
                    <div class="section-header">CLINICAL TRIAGE SUMMARY</div>
                    <p><b>Sample ID:</b> {res['filename']} &nbsp;|&nbsp; <b>Timestamp:</b> {res['timestamp']}</p>
                    <hr style="border: 0.5px solid #E2E8F0;">
                    <h2 style="color: {status_color};">{res['class'].upper()}</h2>
                    <p><b>AI Confidence:</b> {res['conf']:.2f}%</p>
                    <p><b>Triage Status:</b> <span style="color:{status_color}; font-weight:bold;">{res['triage']}</span></p>
                    <br>
                    <div class="section-header">PHYSICIAN CONSULTATION NOTE</div>
                    <p style="font-style: italic; color: #475569; line-height: 1.6;">{res['summary']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br><b>Explainable AI (XAI) Activation Map:</b>", unsafe_allow_html=True)
            st.image(res['heatmap'], caption="Regions of highest pathological significance", width=350)
            
            # FEATURE 3: PDF EXPORT
            pdf_bytes = create_pdf(res)
            st.download_button(
                label="📄 Download Official PDF Dossier",
                data=pdf_bytes,
                file_name=f"Report_{res['filename'].replace('/', '_')}.pdf",
                mime="application/pdf"
            )
        else:
            st.info("Awaiting patient sample for analysis...")

with tab_batch:
    st.subheader("High-Throughput Batch Processing")
    st.write("Upload a batch of `.png` or `.jpg` files to instantly generate a CSV summary report.")
    
    batch_files = st.file_uploader("Upload Multiple Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if batch_files and st.button("Process Batch"):
        progress_bar = st.progress(0)
        batch_results = []
        
        for idx, b_file in enumerate(batch_files):
            b_img = Image.open(b_file).convert('RGB')
            in_tensor = inference_transforms(b_img).unsqueeze(0).to(device)
            with torch.no_grad():
                b_out = clinical_model(in_tensor)
                b_prob = torch.nn.functional.softmax(b_out, dim=1)
                b_conf, b_pred = torch.max(b_prob, 1)
            
            b_class = "Parasitized" if b_pred.item() == 0 else "Uninfected"
            b_conf_val = b_conf.item() * 100
            b_triage = "CLEAR" if b_conf_val >= 85 else "REVIEW REQUIRED"
            
            batch_results.append({
                "Filename": b_file.name,
                "Diagnosis": b_class,
                "Confidence (%)": round(b_conf_val, 2),
                "Triage Status": b_triage
            })
            progress_bar.progress((idx + 1) / len(batch_files))
            
        st.success(f"Successfully processed {len(batch_files)} images!")
        df = pd.DataFrame(batch_results)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Download Batch Report (CSV)", data=csv, file_name="batch_malaria_report.csv", mime="text/csv")

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
