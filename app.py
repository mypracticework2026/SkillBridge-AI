import streamlit as st
import pdfplumber
import pickle
import re
import nltk
import io
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI SkillBridge - CV Screening",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS (Streamlined) ------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    .stApp {
        background: linear-gradient(135deg, #F7F3EA 0%, #EFE7D6 100%);
        font-family: 'Inter', sans-serif;
    }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #1F4D3E 0%, #2E7D5B 100%);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(31, 77, 62, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #F7F3EA;
        position: relative;
        overflow: hidden;
    }
    .hero-text h1 {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 0;
        color: #F7F3EA;
        text-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .hero-illustration {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 0 10px;
    }
    .doc-item {
        font-size: 2.1rem;
        opacity: 0.45;
        filter: grayscale(60%);
        transition: all 0.2s ease;
    }
    .doc-item.highlighted {
        position: relative;
        opacity: 1;
        filter: none;
        transform: scale(1.35) translateY(-6px);
        background: rgba(255,255,255,0.15);
        border-radius: 16px;
        padding: 10px 16px;
        backdrop-filter: blur(4px);
        border: 2px solid rgba(255,255,255,0.5);
        box-shadow: 0 0 30px rgba(255,255,255,0.3);
    }
    .doc-item.highlighted .glass-icon {
        position: absolute;
        top: -16px;
        right: -16px;
        font-size: 1.5rem;
        transform: rotate(15deg);
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.6));
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0% { transform: rotate(10deg) translateY(0px); }
        50% { transform: rotate(20deg) translateY(-8px); }
        100% { transform: rotate(10deg) translateY(0px); }
    }

    /* Upload Cards */
    .upload-card {
        background: #FFFFFF;
        padding: 2rem 1.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(44,42,36,0.05);
        border: 1px solid #E5DCC5;
        transition: transform 0.2s ease;
        height: 100%;
    }
    .upload-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(31, 77, 62, 0.12);
    }
    .stButton button {
        background: linear-gradient(135deg, #1F4D3E 0%, #2E7D5B 100%);
        color: #F7F3EA !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 14px rgba(31, 77, 62, 0.35);
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton button:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 25px rgba(31, 77, 62, 0.5);
    }
    .stRadio > div {
        display: flex;
        gap: 20px;
        background: #FFFFFF;
        padding: 10px 20px;
        border-radius: 40px;
        box-shadow: 0 2px 8px rgba(44,42,36,0.04);
        border: 1px solid #E5DCC5;
    }
    .stRadio label {
        background: transparent;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 500;
        transition: all 0.2s;
    }
    .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #1F4D3E, #2E7D5B);
        color: #F7F3EA !important;
        box-shadow: 0 4px 10px rgba(31, 77, 62, 0.3);
    }
    .result-card {
        background: #FFFFFF;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(44,42,36,0.06);
        border: 1px solid #E5DCC5;
        margin-top: 1.5rem;
    }
    .skill-chip-green {
        background: #E1EDE6;
        color: #1F4D3E;
        padding: 6px 14px;
        border-radius: 30px;
        display: inline-block;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #B9D9C7;
    }
    .skill-chip-red {
        background: #F3E4D8;
        color: #8A4A2D;
        padding: 6px 14px;
        border-radius: 30px;
        display: inline-block;
        margin: 4px 6px 4px 0;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid #E3C2A8;
    }
    .custom-progress {
        height: 12px;
        border-radius: 10px;
        background: #EFE7D6;
        overflow: hidden;
        margin: 0.5rem 0 1rem 0;
    }
    .custom-progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #1F4D3E, #2E7D5B);
        width: 0%;
        transition: width 0.8s ease;
    }
    .match-score-number {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1F4D3E, #2E7D5B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    
    /* Animated Robot */
    .ai-robot-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: rgba(31, 77, 62, 0.03);
        border-radius: 20px;
        padding: 15px 10px;
        border: 1px dashed #2E7D5B;
        height: 100%;
        min-height: 180px;
    }
    .ai-robot { font-size: 4.5rem; animation: float 2.5s ease-in-out infinite; display: inline-block; }
    .ai-robot-resume { font-size: 1.8rem; animation: resume-sway 3s ease-in-out infinite; display: inline-block; margin-left: -10px; }
    @keyframes resume-sway { 0% { transform: rotate(-5deg); } 50% { transform: rotate(10deg); } 100% { transform: rotate(-5deg); } }
    .ai-robot-label { font-size: 0.75rem; font-weight: 600; color: #1F4D3E; background: #E1EDE6; padding: 4px 14px; border-radius: 30px; margin-top: 5px; }

    @media (max-width: 768px) {
        .hero-container { flex-direction: column; text-align: center; }
        .hero-text h1 { font-size: 2.2rem; }
        .hero-illustration { margin-top: 1rem; gap: 8px; }
        .doc-item { font-size: 1.6rem; }
        .doc-item.highlighted { transform: scale(1.2) translateY(-4px); }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ NLTK, MODEL, HELPERS ------------------
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return nltk.corpus.stopwords.words('english')

stop_words = set(load_nltk())

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

# ------------------ HERO ------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-text"><h1>AI SkillBridge</h1></div>
    <div class="hero-illustration">
        <div class="doc-item">📄</div>
        <div class="doc-item">📄</div>
        <div class="doc-item highlighted">
            📄
            <div class="glass-icon">🔍</div>
        </div>
        <div class="doc-item">📄</div>
        <div class="doc-item">📄</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ UPLOAD ------------------
col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload Candidate CV")
    cv_file = st.file_uploader("Choose a PDF", type=['pdf'], key="cv_upload", label_visibility="collapsed")
    if cv_file: st.success(f"✅ {cv_file.name} uploaded")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📌 Upload Job Description")
    jd_file = st.file_uploader("Choose a PDF", type=['pdf'], key="jd_upload", label_visibility="collapsed")
    if jd_file: st.success(f"✅ {jd_file.name} uploaded")
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
                cv_cleaned = wash_text(cv_text)
                jd_cleaned = wash_text(jd_text)

                if cv_cleaned and jd_cleaned:
                    cv_vector = vectorizer.transform([cv_cleaned])
                    jd_vector = vectorizer.transform([jd_cleaned])

                    score = cosine_similarity(cv_vector, jd_vector)[0][0]
                    match_percentage = round(score * 100, 2)

                    jd_array = jd_vector.toarray().flatten()
                    word_score_pairs = sorted(zip(feature_names, jd_array), key=lambda x: x[1], reverse=True)
                    required_skills = [word for word, w_score in word_score_pairs if w_score > 0][:15]

                    cv_words = set(cv_cleaned.split())
                    matched = [skill for skill in required_skills if skill in cv_words]
                    missing = [skill for skill in required_skills if skill not in cv_words]

                    # ====================== RESULTS DISPLAY ======================
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)

                    # Row 1: Score + Skills
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.markdown(f"<div class='match-score-number'>{match_percentage}%</div>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="custom-progress">
                            <div class="custom-progress-fill" style="width: {min(match_percentage, 100)}%;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        mode_str = "Candidate" if "Candidate" in mode else "Recruiter"
                        if mode_str == "Candidate":
                            if match_percentage >= 80: st.success("🌟 Excellent Match!")
                            elif match_percentage >= 50: st.info("📈 Good Foundation")
                            else: st.warning("🔄 Consider Upskilling")
                        else:
                            if match_percentage >= 70: st.success("📞 Shortlist")
                            elif match_percentage >= 40: st.warning("🧐 Review")
                            else: st.error("❌ Reject")

                    with res_col2:
                        st.markdown(f"**✅ Matched Skills ({len(matched)})**")
                        match_html = "".join([f"<span class='skill-chip-green'>{s}</span>" for s in matched]) if matched else "None"
                        st.markdown(match_html, unsafe_allow_html=True)
                        st.markdown(f"**❌ Missing Skills ({len(missing)})**")
                        miss_html = "".join([f"<span class='skill-chip-red'>{s}</span>" for s in missing]) if missing else "None"
                        st.markdown(miss_html, unsafe_allow_html=True)

                    st.divider()

                    # ====================== ROW 2: AI ROBOT + FIXED ADVICE ======================
                    advice_col1, advice_col2 = st.columns([1, 3], gap="medium")

                    # Left: Robot
                    with advice_col1:
                        st.markdown("""
                        <div class="ai-robot-container">
                            <div><span class="ai-robot">🤖</span><span class="ai-robot-resume">📄</span></div>
                            <div class="ai-robot-label">🔍 AI Scanning</div>
                            <div style="font-size:0.7rem; color:#6B6656; margin-top:5px;">Analyzing your resume...</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Right: Eye-Catchy Advice Box (FIXED with inline styles to prevent raw HTML)
                    with advice_col2:
                        if missing:
                            # Build the list items dynamically with pure inline styles (no global classes needed)
                            list_items = ""
                            for idx, skill in enumerate(missing[:7], start=1):
                                list_items += f"""
                                <li style="display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #F0EBE0; font-size: 0.95rem; color: #2C2A24; list-style: none;">
                                    <span style="background: #1F4D3E; color: #F7F3EA; font-weight: 700; font-size: 0.75rem; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; flex-shrink: 0;">{idx}</span>
                                    <span style="flex: 1;">Gap detected in <strong>'{skill}'</strong></span>
                                    <span style="background: #F3E4D8; color: #8A4A2D; padding: 2px 12px; border-radius: 30px; font-weight: 500; font-size: 0.75rem; white-space: nowrap;">Priority {idx}</span>
                                </li>
                                """

                            if mode_str == "Candidate":
                                title = "📌 Personalized Upskilling Plan"
                                action_msg = "📚 Consider taking relevant courses or building projects to bridge these gaps."
                            else:
                                title = "📌 Hiring Recommendation (Skill Gaps)"
                                action_msg = "🧐 Verify if these missing skills are strictly required for the role."

                            advice_html = f"""
                            <div style="background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.8rem; border-left: 6px solid #2E7D5B; box-shadow: 0 8px 24px rgba(31, 77, 62, 0.08); height: 100%; min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #1F4D3E; font-weight: 700; margin-top: 0; margin-bottom: 10px; font-size: 1.1rem; display: flex; align-items: center; gap: 8px;">{title}</h4>
                                <ul style="list-style: none; padding: 0; margin: 0;">
                                    {list_items}
                                </ul>
                                <div style="margin-top: 12px; font-size: 0.9rem; color: #1F4D3E; background: #E1EDE6; padding: 8px 16px; border-radius: 30px; display: inline-block; font-weight: 500; align-self: flex-start;">💡 {action_msg}</div>
                            </div>
                            """
                        else:
                            # Perfect match
                            if mode_str == "Candidate":
                                main_text = "🌟 Perfect Match! You cover all the key requirements. Proceed with confidence!"
                            else:
                                main_text = "🌟 Perfect Match! This candidate covers all key requirements."
                            
                            advice_html = f"""
                            <div style="background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.8rem; border-left: 6px solid #4CAF50; box-shadow: 0 8px 24px rgba(31, 77, 62, 0.08); height: 100%; min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #1F4D3E; font-weight: 700; margin-top: 0; font-size: 1.1rem;">🎯 Verdict</h4>
                                <p style="font-size:1.1rem; font-weight:500; color:#1F4D3E;">{main_text}</p>
                            </div>
                            """
                        
                        st.markdown(advice_html, unsafe_allow_html=True)

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
    "<p style='text-align: center; color: #6B6656; font-size: 0.8rem;'>Built with ❤️ using Streamlit, Scikit-Learn, and PDFPlumber</p>",
    unsafe_allow_html=True
)
