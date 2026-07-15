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

# ------------------ CUSTOM CSS (with changes) ------------------
st.markdown("""
<style>
    /* Hide default Streamlit elements */
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

    /* ----- HERO SECTION (Green box) ----- */
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
        letter-spacing: -1px;
        color: #F7F3EA;
        text-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    /* Row of resumes with one highlighted under the magnifying glass */
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

    /* Buttons */
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

    /* Radio Buttons (Mode Toggle) */
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

    /* Results Cards */
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

    /* Custom Progress Bar */
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

# ================== MODIFIED HERO (no branding bar) ==================
st.markdown("""
<div class="hero-container">
    <div class="hero-text">
        <h1>AI SkillBridge</h1>
        <!-- subtitle and 'powered by' removed -->
    </div>
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

# ------------------ UPLOAD, MODE, ANALYZE (unchanged) ------------------
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

                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
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

st.divider()
st.markdown(
    "<p style='text-align: center; color: #6B6656; font-size: 0.8rem;'>Built with ❤️ using Streamlit, Scikit-Learn, and PDFPlumber</p>",
    unsafe_allow_html=True
)
