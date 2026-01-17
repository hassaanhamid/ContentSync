"""
ContentSync Pro - Multimodal AI Content-Image Alignment System
Author: Hassaan Hamid Okarvi (FA22-BSCS-0060)
7th Semester Computer Science Portfolio Project
"""

import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# PAGE CONFIGURATION
st.set_page_config(
    page_title="ContentSync Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS FOR PAGE
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h1 {
        color: #ffffff;
        font-weight: 800;
    }
    
    h2, h3 {
        color: #e0e0e0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# MODEL LOAD
@st.cache_resource
def load_clip_model():
    """Load CLIP model"""
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor
    except Exception as e:
        st.error(f"❌ Model Loading Failed: {str(e)}")
        return None, None

# TEXT CHUNKING
def chunk_text(text: str, max_tokens: int = 75) -> list:
    """
    Split text into chunks that fit CLIP's 77-token limit.
    Uses sentence boundaries for natural splits.
    """
    # Split by sentences
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Rough estimate: 1 token ≈ 4 characters
        if len(current_chunk) + len(sentence) < max_tokens * 4:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks if chunks else [text[:300]]  # Fallback

# ALIGNMENT
def calculate_alignment(image: Image.Image, text: str, clip_model, clip_processor) -> float:
    """
    Calculate alignment by processing text in chunks and averaging scores.
    Returns calibrated 0-100% score.
    """
    try:
        # Split text into CLIP-compatible chunks
        text_chunks = chunk_text(text)
        
        all_scores = []
        
        for chunk in text_chunks:
            inputs = clip_processor(
                text=[chunk],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logit = outputs.logits_per_image[0][0].item()
                all_scores.append(logit)
        
        # Average all chunk scores
        avg_logit = np.mean(all_scores)
        max_logit = np.max(all_scores)
        
        # CALIBRATED SCORING
        # Based on extensive CLIP testing:
        # Unrelated: 15-20, Weak: 20-23, Moderate: 23-26, Good: 26-29, Excellent: 29-33
        
        # Use weighted combination of average and max
        combined_logit = (avg_logit * 0.7) + (max_logit * 0.3)
        
        # Map to 0-100 scale with proper calibration
        MIN_LOGIT = 17.0
        MAX_LOGIT = 30.0
        
        score = ((combined_logit - MIN_LOGIT) / (MAX_LOGIT - MIN_LOGIT)) * 100
        
        # Apply sigmoid-like curve for better separation
        if score < 40:
            score = score * 0.7  # Penalize poor matches more
        elif score > 65:
            score = 65 + (score - 65) * 1.3  # Reward good matches more
        
        # Clamp to [0, 100]
        score = max(0, min(100, score))
        
        return round(score, 1)
        
    except Exception as e:
        st.error(f"❌ Alignment Calculation Failed: {str(e)}")
        return 0.0

# MAIN APPLICATION
def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>ContentSync</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888; font-size: 1.1rem;'>AI-Powered Content-Image Alignment Analysis</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 System Status")
        
        with st.spinner("Loading CLIP Model..."):
            clip_model, clip_processor = load_clip_model()
        
        if clip_model:
            st.success("✅ Model Ready")
        
        st.markdown("---")
        st.markdown("### 👨‍💻 Developer")
        st.markdown("**Hassaan Hamid Okarvi**")
        st.markdown("FA22-BSCS-0060")
        st.markdown("Machine Learning Project")
    
    # Main Content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Article Content")
        article_text = st.text_area(
            "Paste your article here",
            height=350,
            placeholder="Enter your article text..."
        )
        
        st.markdown("### 🖼️ Upload Image")
        uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Alignment Score")
        
        if st.button("🚀 Analyze", use_container_width=True, type="primary"):
            if not article_text or not uploaded_file:
                st.error("⚠️ Please provide both text and image")
            elif not clip_model:
                st.error("⚠️ Model not loaded")
            else:
                with st.spinner("Analyzing alignment..."):
                    score = calculate_alignment(image, article_text, clip_model, clip_processor)
                
                # Display Score
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric(
                    label="",
                    value=f"{score}%"
                )
                
                # Interpretation
                st.markdown("<br>", unsafe_allow_html=True)
                if score >= 70:
                    st.success("### ✅ Excellent Match\nImage strongly aligns with content")
                elif score >= 50:
                    st.warning("### ⚠️ Moderate Match\nImage partially aligns with content")
                elif score >= 30:
                    st.error("### ❌ Poor Match\nImage weakly aligns with content")
                else:
                    st.error("### ❌ Very Poor Match\nImage does not align with content")

if __name__ == "__main__":
    main()