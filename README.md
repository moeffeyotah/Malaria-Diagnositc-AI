Technical Report: Automated Malaria Diagnostic Pipeline
A Clinical Decision Support System (CDSS) Bridging Deep Learning and Generative AI

Author: Moses Mudiaga Effeyotah
Course: INFO 6147 – Deep Learning with PyTorch (Capstone Project)
Date: April 2026

1. Executive Summary
This report details the architecture, training, and deployment of a Clinical Decision Support System (CDSS) engineered to automate the detection of Plasmodium parasites in peripheral blood smears. By bridging a fine-tuned Convolutional Neural Network (CNN) with a state-of-the-art Large Language Model (LLM), this pipeline not only provides highly accurate diagnostic classification but also generates structured, physician-level pathology reports. The system is deployed as a secure, interactive web application utilizing a "Swiss Medical" User Interface standard.

2. Vision Architecture: ResNet-50
The core computer vision engine is built upon the PyTorch framework, utilizing the ResNet-50 architecture. ResNet-50 was selected due to its deep residual learning framework, which effectively mitigates the vanishing gradient problem while capturing the microscopic morphological nuances of intracellular parasitic inclusions.

2.1 Modification and Transfer Learning
To optimize the network for binary classification (Parasitized vs. Uninfected), the standard 1000-class ImageNet fully connected head was truncated. A custom sequential classifier was engineered:

Linear(in_features, 256)

ReLU() Activation

Dropout(p=0.5) to aggressively prevent overfitting on the cellular dataset.

Linear(256, 2) for final unnormalized logits.

2.2 Data Integrity and Transformations
Standardizing the input tensors is critical for clinical reliability. All uploaded macroscopic images undergo a strict deterministic transformation pipeline:

Resizing: (224, 224) to match the expected ResNet-50 input dimensionality.

Tensor Conversion: Normalizing pixel intensities to [0.0, 1.0].

Color Normalization: Applied using standard ImageNet mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225] to align the blood smear chromaticity with the pre-trained feature maps.

3. Clinical Synthesis Engine: Gemini 2.5 Flash
A critical limitation of traditional diagnostic AI is the "black box" delivery of binary results, which lacks clinical utility. To bridge this gap, this system integrates the Google GenAI SDK, leveraging the high-speed gemini-2.5-flash model.

3.1 Prompt Engineering and Routing
The LLM is strictly constrained via prompt engineering to act as an "Expert Clinical Hematologist." By passing the ResNet-50 outputs (Diagnosis and Confidence Percentage) directly into the LLM context window, the model generates a dynamic, highly contextualized report. The prompt enforces:

Medical Terminology: Usage of terms like "erythrocytic morphology" and "leukocytosis."

Doctor-to-Doctor Tone: Stripping all conversational AI filler (e.g., "Certainly! I can help with that.").

Protocol Recommendations: Suggesting actionable next steps, such as serial smears or Complete Blood Counts (CBC).

4. Front-End Deployment and UI/UX
The application is deployed via Streamlit, serving as the presentation layer.

To elevate the application from a standard analytical script to a premium SaaS dashboard, extensive custom CSS was injected. The UI features a "Parchment and Slate" aesthetic, asymmetric column layouts for visual data balancing, and custom metric cards for high-contrast data delivery. The final LLM output is rendered within a simulated "Official Pathology Report" container, complete with automated timestamps and digital signatures, establishing immediate trust and authority with the end-user.

5. Conclusion
This Capstone project successfully demonstrates the viability of multi-modal AI in healthcare environments. By combining PyTorch-driven computer vision with GenAI-driven clinical synthesis, the resulting CDSS provides both rapid diagnostic screening and actionable clinical narratives, significantly reducing the cognitive load on attending hematopathologists.
