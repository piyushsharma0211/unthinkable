import os
import numpy as np
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch
from io import BytesIO
import certifi
import urllib.request
import base64


# ===============================
# CONFIG
# ===============================
DB_PATH = "db"
os.environ['SSL_CERT_FILE'] = certifi.where()

st.set_page_config(
    page_title="Visual Product Matcher", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ===============================
# CUSTOM CSS STYLING
# ===============================
def load_css():
    st.markdown("""
    <style>
    body {
        background: #f4f7f9;
        font-family: 'Poppins', sans-serif;
    }

    .main {
        padding: 2rem;
        color: #222222;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        color: #1a202c;
        background: linear-gradient(to right, #f43f5e, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        font-weight: 400;
        color: #4a5568;
        margin-bottom: 2rem;
    }

    .upload-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.05);
        margin-bottom: 2rem;
    }

    .upload-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin-bottom: 1rem;
    }

    .success-message {
        background: #10b981;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 500;
        margin: 1rem 0;
    }

    .find-button {
        background: linear-gradient(to right, #6366f1, #3b82f6);
        color: white;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.3);
    }

    .find-button:hover {
        transform: scale(1.03);
    }

    .query-image-container {
        background: #fff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e5e7eb;
        margin-top: 1rem;
    }

    .results-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin: 2rem 0 1rem;
    }

    .product-card {
        background: white;
        padding: 1.2rem;
        border-radius: 15px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: 0.3s ease;
    }

    .product-card:hover {
        transform: translateY(-4px);
    }

    .product-image {
        border-radius: 12px;
        width: 100%;
        height: 280px;
        object-fit: cover;
        margin-bottom: 1rem;
    }

    .similarity-score {
        background: #3b82f6;
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        display: inline-block;
        font-weight: 500;
    }

    .custom-divider {
        height: 2px;
        background: linear-gradient(to right, #f43f5e, #6366f1);
        border: none;
        margin: 2rem 0;
    }

    .stFileUploader > div > div > div > div {
        background-color: #f9fafb;
        border: 2px dashed #cbd5e1;
        border-radius: 10px;
        padding: 1.5rem;
        color: #000000;
    }

    .footer-text {
        color: #718096;
        font-size: 0.9rem;
        text-align: center;
        margin-top: 2rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)



# ===============================
# MODEL LOADING
# ===============================
@st.cache_resource
def load_clip_model():
    return SentenceTransformer('clip-ViT-B-32')


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features_clip(image_source, model):
    try:
        if isinstance(image_source, BytesIO):
            image = Image.open(image_source).convert('RGB')
        elif isinstance(image_source, str) and image_source.startswith("http"):
            response = urllib.request.urlopen(image_source)
            image = Image.open(BytesIO(response.read())).convert('RGB')
        else:
            image = Image.open(image_source).convert('RGB')

        emb = model.encode([image], convert_to_tensor=True, show_progress_bar=False)
        return emb
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None


# ===============================
# DATABASE FEATURE VECTORS
# ===============================
@st.cache_data(show_spinner=False)
def get_feature_vectors_from_db(_model, db_path):
    image_paths = []
    feature_vectors = []

    for file in os.listdir(db_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(db_path, file)
            emb = extract_features_clip(img_path, _model)
            if emb is not None:
                feature_vectors.append(emb)
                image_paths.append(img_path)

    if feature_vectors:
        feature_vectors = torch.vstack(feature_vectors)
    else:
        feature_vectors = torch.empty(0)

    return feature_vectors, image_paths


# ===============================
# FIND SIMILAR IMAGES
# ===============================
def find_similar_images(query_emb, feature_vectors, image_paths, top_n=6):
    if feature_vectors.shape[0] == 0:
        st.error("Database is empty! Please add some images.")
        return []

    cosine_scores = util.cos_sim(query_emb, feature_vectors)[0]
    top_results = torch.topk(cosine_scores, k=top_n)

    similar_images = []
    for score, idx in zip(top_results.values, top_results.indices):
        similar_images.append((image_paths[idx], float(score)))

    return similar_images


# ===============================
# UTILITY FUNCTION: Image → Base64
# ===============================
def image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# ===============================
# STREAMLIT UI
# ===============================
def main():
    # Load custom CSS
    load_css()
    
    # Header section
    st.markdown("""
        <h1 class="main-header"> Visual Product Matcher </h1>
        <p class="sub-header">Find matching fashion & product styles by simply uploading an image. Powered by AI magic.</p>
    """, unsafe_allow_html=True)

    # Load model
    model = load_clip_model()

    # Initialize database features
    if "feature_vectors" not in st.session_state:
        with st.spinner("🔄 Extracting database features..."):
            st.session_state.feature_vectors, st.session_state.image_paths = get_feature_vectors_from_db(model, DB_PATH)
        st.success("✅ Database loaded successfully!")

    # Upload section
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<h3 class="upload-header">📤 Upload Your Product Image</h3>', unsafe_allow_html=True)
    
    # Create two columns for upload options
    col1, col2 = st.columns(2)
    
    uploaded_file = None
    url_input = None
    
    with col1:
        st.markdown('<p style="color: #000000;"><strong>Upload from Device</strong></p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    with col2:
        st.markdown('<p style="color: #000000;"><strong>Upload from URL</strong></p>', unsafe_allow_html=True)
        url_input = st.text_input("Paste image URL here", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Process uploaded image
    query_emb = None
    query_image = None
    image_uploaded = False

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file).convert('RGB')
        query_emb = extract_features_clip(uploaded_file, model)
        image_uploaded = True
        
        # Success message
        st.markdown("""
            <div class="success-message">
                ✅ Image successfully uploaded! Ready to find similar products.
            </div>
        """, unsafe_allow_html=True)

    elif url_input:
        try:
            response = urllib.request.urlopen(url_input)
            query_image = Image.open(BytesIO(response.read())).convert('RGB')
            query_emb = extract_features_clip(url_input, model)
            image_uploaded = True
            
            # Success message
            st.markdown("""
                <div class="success-message">
                    ✅ Image from URL successfully loaded! Ready to find similar products.
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"❌ Failed to load image from URL: {e}")

    # Display uploaded image
    if query_image is not None:
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
        
        st.markdown('<div class="query-image-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #000000; margin-bottom: 1rem;">🔍 Your Uploaded Image</h3>', unsafe_allow_html=True)
        
        # Center the image
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(query_image, use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Number of results slider
        st.markdown('<div style="margin: 2rem 0; color: #000000;">', unsafe_allow_html=True)
        top_n = st.slider("🎯 Number of similar products to find", 2, 10, 4)
        st.markdown('</div>', unsafe_allow_html=True)

        # Find similar button
        if st.button("🔍 Find Similar Products", use_container_width=True) and query_emb is not None:
            with st.spinner("🔍 Searching for similar products..."):
                results = find_similar_images(
                    query_emb,
                    st.session_state.feature_vectors,
                    st.session_state.image_paths,
                    top_n
                )

            if results:
                st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
                st.markdown('<h2 class="results-header">✨ Similar Products Found</h2>', unsafe_allow_html=True)

                # Display results in 2-column layout
                for i in range(0, len(results), 2):
                    cols = st.columns(2)
                    
                    # First product in row
                    if i < len(results):
                        img_path, score = results[i]
                        with cols[0]:
                            st.markdown(f"""
                                <div class="product-card">
                                    <img src="data:image/png;base64,{image_to_base64(img_path)}" class="product-image">
                                    <div class="similarity-score">
                                        🎯 Similarity: {score:.1%}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Second product in row (if exists)
                    if i + 1 < len(results):
                        img_path, score = results[i + 1]
                        with cols[1]:
                            st.markdown(f"""
                                <div class="product-card">
                                    <img src="data:image/png;base64,{image_to_base64(img_path)}" class="product-image">
                                    <div class="similarity-score">
                                        🎯 Similarity: {score:.1%}
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No similar products found. Try uploading a different image.")

    # Footer
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align: center; margin-top: 2rem;" class="footer-text">
            <p style="color: #000000;">🚀 Created by <strong>Piyush Kumar</strong> 2025Powered by CLIP Vision Transformer | Built with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)


# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    main()