import streamlit as st
import pdfplumber
import pickle
import re
import nltk
import io
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG (Must be first) ------------------
st.set_page_config(
    page_title="AI SkillBridge - CV Screening",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS INJECTION ------------------
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Global Font & Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9edf5 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }

    /* ----- HERO SECTION (The "CV & Glass" Search) ----- */
    .hero-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .hero-text h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .hero-text p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 0.2rem;
        font-weight: 300;
    }
    
    /* The "Magnifying Glass over CV" Illustration (Pure CSS/HTML) */
    .hero-illustration {
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        width: 150px;
        height: 150px;
        background: rgba(255,255,255,0.15);
        border-radius: 50%;
        backdrop-filter: blur(5px);
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 0 30px rgba(255,255,255,0.2);
    }
    .hero-illustration .doc-icon {
        font-size: 4.5rem;
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.3));
        position: relative;
        right: -10px;
    }
    .hero-illustration .glass-icon {
        font-size: 3.5rem;
        position: absolute;
        top: -5px;
        right: -5px;
        transform: rotate(15deg);
        filter: drop-shadow(0 0 15px rgba(255,215,0,0.6));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: rotate(10deg) translateY(0px); }
        50% { transform: rotate(20deg) translateY(-10px); }
        100% { transform: rotate(10deg) translateY(0px); }
    }

    /* Branding Top Left */
    .branding {
        position: sticky;
        top: 0;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
        padding: 0.8rem 2rem;
        border-bottom: 1px solid rgba(0,0,0,0.05);
        z-index: 999;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 700;
        font-size: 1.5rem;
        color: #4a3f5c;
        margin-bottom: 1rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.02);
    }
    .branding span {
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Upload Cards */
    .upload-card {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.02);
        transition: transform 0.2s ease;
        height: 100%;
    }
    .upload-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(102, 126, 234, 0.10);
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }

    /* Radio Buttons (Mode Toggle) */
    .stRadio > div {
        display: flex;
        gap: 20px;
        background: white;
        padding: 10px 20px;
        border-radius: 40px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        border: 1px solid #f0f0f5;
    }
    .stRadio label {
        background: transparent;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white !important;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }

    /* Results Cards */
    .result-card {
        background: white;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        border: 1px solid rgba(255,255,255,0.5);
        margin-top: 1.5rem;
    }
    .skill-chip-green {
        background: #e8f5e9;
        color: #2e7d32;
        padding: 6px 14px;
        border-radius: 30px;
        display: inline-block;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #a5d6a7;
    }
    .skill-chip-red {
        background: #ffebee;
        color: #c62828;
        padding: 6px 14px;
        border-radius: 30px;
        display: inline-block;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #ef9a9a;
    }

    /* Custom Progress Bar */
    .custom-progress {
        height: 12px;
        border-radius: 10px;
        background: #e9ecef;
        overflow: hidden;
        margin: 0.5rem 0 1rem 0;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        width: 0%;
        transition: width 0.8s ease;
    }

    .match-score-number {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .hero-container { flex-direction: column; text-align: center; }
        .hero-text h1 { font-size: 2.2rem; }
        .hero-illustration { width: 100px; height: 100px; margin-top: 1rem; }
        .hero-illustration .doc-icon { font-size: 3rem; }
        .hero-illustration .glass-icon { font-size: 2.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ DOWNLOAD NLTK DATA ------------------
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return nltk.corpus.stopwords.words('english')

stop_words = set(load_nltk())

# ------------------ LOAD THE SAVED MODEL ------------------
@st.cache_resource
def load_model():
    try:
        with open('aicvscreening_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("🚨 Model file 'aicvscreening_model.pkl' not found!")
        st.stop()

model = load_model()
vectorizer = model['vectorizer']
feature_names = vectorizer.get_feature_names_out()

# ------------------ TEXT CLEANING ------------------
def wash_text(raw_text):
    if not raw_text:
        return ""
    text = raw_text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(cleaned_words)

def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.getbuffer())) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() or ""
        return full_text
    except:
        return ""

# ------------------ TOP LEFT BRANDING ------------------
st.markdown("""
<div class="branding">
    🔍 <span>AI SkillBridge</span> 
    <span style="font-size:0.8rem; font-weight:400; color:gray; margin-left:auto;">v1.0</span>
</div>
""", unsafe_allow_html=True)

# ------------------ HERO SECTION (CV + Glass) ------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-text">
        <h1>🤖 AI CV Screening</h1>
        <p>Intelligent Resume vs Job Description Matcher</p>
        <p style="font-size:0.9rem; opacity:0.7; margin-top:0.5rem;">
            📊 Powered by TF-IDF & Cosine Similarity
        </p>
    </div>
    <div class="hero-illustration">
        <div class="doc-icon">📄</div>
        <div class="glass-icon">🔍</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ UPLOAD SECTION ------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Candidate CV")
    cv_file = st.file_uploader("Choose a PDF", type=['pdf'], key="cv_upload", label_visibility="collapsed")
    if cv_file:
        st.success(f"✅ {cv_file.name} uploaded")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📌 Upload Job Description")
    jd_file = st.file_uploader("Choose a PDF", type=['pdf'], key="jd_upload", label_visibility="collapsed")
    if jd_file:
        st.success(f"✅ {jd_file.name} uploaded")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ MODE & ANALYZE ------------------
mode = st.radio(
    "👤 Select View",
    ["🎓 Candidate Mode (Upskilling Advice)", "💼 Recruiter Mode (Hiring Decision)"],
    horizontal=True,
    index=0
)

if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    if cv_file and jd_file:
        with st.spinner("🔍 Analyzing..."):
            cv_text = extract_text_from_pdf(cv_file)
            jd_text = extract_text_from_pdf(jd_file)
            
            if cv_text and jd_text:
                # Clean texts
                cv_cleaned = wash_text(cv_text)
                jd_cleaned = wash_text(jd_text)
                
                if cv_cleaned and jd_cleaned:
                    # Transform
                    cv_vector = vectorizer.transform([cv_cleaned])
                    jd_vector = vectorizer.transform([jd_cleaned])
                    
                    # Score
                    score = cosine_similarity(cv_vector, jd_vector)[0][0]
                    match_percentage = round(score * 100, 2)
                    
                    # Extract skills from JD
                    jd_array = jd_vector.toarray().flatten()
                    word_score_pairs = sorted(zip(feature_names, jd_array), key=lambda x: x[1], reverse=True)
                    required_skills = [word for word, w_score in word_score_pairs if w_score > 0][:15]
                    
                    cv_words = set(cv_cleaned.split())
                    matched = [skill for skill in required_skills if skill in cv_words]
                    missing = [skill for skill in required_skills if skill not in cv_words]
                    
                    # --- DISPLAY RESULTS IN A BEAUTIFUL CARD ---
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # Row 1: Score + Skills
                    res_col1, res_col2 = st.columns([1, 2])
                    
                    with res_col1:
                        st.markdown(f"<div class='match-score-number'>{match_percentage}%</div>", unsafe_allow_html=True)
                        # Custom Progress Bar
                        st.markdown(f"""
                        <div class="custom-progress">
                            <div class="custom-progress-fill" style="width: {min(match_percentage, 100)}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        mode_str = "Candidate" if "Candidate" in mode else "Recruiter"
                        if mode_str == "Candidate":
                            if match_percentage >= 80:
                                st.success("🌟 Excellent Match!")
                            elif match_percentage >= 50:
                                st.info("📈 Good Foundation")
                            else:
                                st.warning("🔄 Consider Upskilling")
                        else:
                            if match_percentage >= 70:
                                st.success("📞 Shortlist")
                            elif match_percentage >= 40:
                                st.warning("🧐 Review")
                            else:
                                st.error("❌ Reject")
                    
                    with res_col2:
                        st.markdown(f"**✅ Matched Skills ({len(matched)})**")
                        match_html = "".join([f"<span class='skill-chip-green'>{s}</span>" for s in matched]) if matched else "None"
                        st.markdown(match_html, unsafe_allow_html=True)
                        
                        st.markdown(f"**❌ Missing Skills ({len(missing)})**")
                        miss_html = "".join([f"<span class='skill-chip-red'>{s}</span>" for s in missing]) if missing else "None"
                        st.markdown(miss_html, unsafe_allow_html=True)
                    
                    # Row 2: Advice
                    st.divider()
                    if mode_str == "Candidate":
                        if match_percentage >= 80:
                            advice = "🌟 EXCELLENT FIT! Highlight these matched skills in your interview."
                        elif match_percentage >= 50:
                            advice = f"📈 Good foundation. Focus on: {', '.join(missing[:3])}. Consider a certification."
                        elif match_percentage >= 30:
                            advice = f"⚠️ Significant gaps. Focus on: {', '.join(missing[:5])}. Take relevant courses."
                        else:
                            advice = f"🔄 Career pivot needed. Missing: {', '.join(missing[:5])}. Explore junior roles."
                        st.info(f"💡 **Advice:** {advice}")
                    else:
                        if match_percentage >= 70:
                            decision = "📞 SHORTLIST: Proceed to interview."
                        elif match_percentage >= 40:
                            decision = f"🧐 REVIEW: Check missing: {', '.join(missing[:3])}."
                        else:
                            decision = f"❌ REJECT: Not suitable. Missing: {', '.join(missing[:5])}."
                        st.info(f"⚖️ **Decision:** {decision}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Could not extract meaningful text.")
            else:
                st.error("Failed to extract text from PDFs.")
    else:
        st.warning("⚠️ Please upload both a CV and a Job Description.")

# ------------------ FOOTER ------------------
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8rem;'>Built with ❤️ using Streamlit, Scikit-Learn, and PDFPlumber</p>",
    unsafe_allow_html=True
)
