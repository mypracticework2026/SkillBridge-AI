import streamlit as st
import pdfplumber
import pickle
import re
import nltk
import io
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI SkillBridge - Smart CV Screening",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    .stApp { background: #f0f4f8; font-family: 'Inter', sans-serif; }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1200px;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* ----- HEADER / NAVBAR ----- */
    .navbar {
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(10px);
        padding: 0.7rem 2rem;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        margin-bottom: 1.5rem;
        border-radius: 0 0 20px 20px;
    }
    .navbar-left { display: flex; align-items: center; gap: 20px; font-size: 1rem; color: #0f172a; }
    .navbar-left a { text-decoration: none; color: #0f172a; display: flex; align-items: center; gap: 6px; font-weight: 500; transition: color 0.2s; }
    .navbar-left a:hover { color: #0d9488; }
    .navbar-right { display: flex; align-items: center; gap: 25px; font-weight: 500; }
    .navbar-right a { text-decoration: none; color: #334155; transition: color 0.2s; }
    .navbar-right a:hover { color: #0d9488; }
    .btn-outline { border: 2px solid #0d9488; padding: 6px 18px; border-radius: 40px; color: #0d9488 !important; font-weight: 600; }
    .btn-outline:hover { background: #0d9488; color: white !important; }

    /* ----- HERO SECTION ----- */
    .hero {
        background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 50%, #a7f3d0 100%);
        border-radius: 30px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 20px 60px rgba(13, 148, 136, 0.2);
        min-height: 260px;
        flex-wrap: wrap;
    }
    .hero-left { flex: 1 1 300px; color: #0f172a; }
    .hero-left h1 { font-size: 3.2rem; font-weight: 800; margin: 0; line-height: 1.1; color: #0f172a; }
    .hero-left p { font-size: 1.2rem; margin: 0.5rem 0 0 0; opacity: 0.85; }
    .hero-right { display: flex; align-items: center; justify-content: center; gap: 15px; padding: 10px; flex-wrap: wrap; }
    .cv-icon { font-size: 2.8rem; opacity: 0.5; transition: transform 0.3s; }
    .cv-icon.highlighted {
        font-size: 4.2rem;
        opacity: 1;
        transform: scale(1.2) translateY(-8px);
        background: rgba(255,255,255,0.25);
        border-radius: 16px;
        padding: 0 10px;
        backdrop-filter: blur(4px);
        border: 2px solid rgba(255,255,255,0.6);
        box-shadow: 0 0 30px rgba(255,255,255,0.3);
        position: relative;
    }
    .glass-icon {
        position: absolute;
        top: -12px;
        right: -12px;
        font-size: 2rem;
        transform: rotate(20deg);
        filter: drop-shadow(0 0 10px rgba(255,255,255,0.8));
        animation: float-glass 3s ease-in-out infinite;
    }
    @keyframes float-glass {
        0% { transform: rotate(15deg) translateY(0); }
        50% { transform: rotate(25deg) translateY(-10px); }
        100% { transform: rotate(15deg) translateY(0); }
    }

    /* ----- UPLOAD CARDS ----- */
    .upload-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
        transition: border-color 0.2s;
        height: 100%;
    }
    .upload-card:hover { border-color: #0d9488; }

    /* ----- RESULT BOXES (Dotted Border) ----- */
    .result-box {
        background: white;
        border: 3px dotted #0d9488;
        border-radius: 20px;
        padding: 1.5rem 1.2rem;
        height: 100%;
        min-height: 180px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.02);
    }
    .result-box h4 {
        margin: 0 0 0.8rem 0;
        color: #0f172a;
        font-weight: 700;
        font-size: 1.1rem;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 8px;
    }
    .chip {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 6px 4px 0;
    }
    .chip-green { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .chip-red { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
    .chip-neutral { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

    /* ----- ADVICE BOX ----- */
    .advice-box {
        background: #f8fafc;
        border-radius: 16px;
        padding: 1.2rem 1.8rem;
        border-left: 6px solid #0d9488;
        margin: 1.2rem 0;
        font-size: 1rem;
    }

    /* ----- FOOTER (Now with Hero Colors) ----- */
    .footer {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white !important;
        padding: 2rem 2rem;
        border-radius: 20px 20px 0 0;
        margin-top: 3rem;
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1.5rem;
        align-items: center;
    }
    .footer a {
        color: #ccfbf1;
        text-decoration: none;
        transition: color 0.2s;
        margin-left: 1.5rem;
    }
    .footer a:hover { color: white; }
    .footer-left { font-weight: 700; font-size: 1.1rem; }

    /* Buttons & Radio */
    .stButton button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35);
        width: 100%;
        transition: transform 0.2s;
    }
    .stButton button:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(13, 148, 136, 0.5); }
    .stRadio > div {
        display: flex;
        gap: 20px;
        background: white;
        padding: 10px 20px;
        border-radius: 40px;
        border: 1px solid #e2e8f0;
    }
    .stRadio label[data-baseweb="radio"]:has(input:checked) {
        background: #0d9488;
        color: white !important;
        border-radius: 30px;
        padding: 8px 20px;
    }
    @media (max-width: 768px) {
        .hero { flex-direction: column; text-align: center; }
        .hero-left h1 { font-size: 2.2rem; }
        .navbar-right { display: none; }
        .footer { flex-direction: column; text-align: center; }
        .footer a { margin: 0 0.5rem; }
    }
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD NLTK & MODEL ------------------
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

# ------------------ HEADER / NAVBAR ------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-left">
        <a href="#"><span>📞</span> +1 (555) 123-4567</a>
        <a href="#"><span>📱</span> WhatsApp</a>
        <a href="#"><span>✉️</span> hello@aiskillbridge.com</a>
    </div>
    <div class="navbar-right">
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#">Pricing</a>
        <a href="#" class="btn-outline">Get Started</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ HERO SECTION ------------------
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <h1>AI SkillBridge</h1>
        <p>Find the perfect match between resumes and job descriptions with AI-powered insights.</p>
    </div>
    <div class="hero-right">
        <span class="cv-icon">📄</span>
        <span class="cv-icon">📄</span>
        <span class="cv-icon highlighted">
            📄
            <span class="glass-icon">🔍</span>
        </span>
        <span class="cv-icon">📄</span>
        <span class="cv-icon">📄</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ UPLOAD SECTION ------------------
st.markdown("### 🚀 Live Demo – Upload & Analyze")

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
        with st.spinner("🔍 AI is analyzing..."):
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
                    
                    mode_str = "Candidate" if "Candidate" in mode else "Recruiter"

                    # --- DISPLAY RESULTS ---
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")

                    # Row 1: Two columns for Matched and Gaps
                    col_a, col_b = st.columns(2, gap="large")

                    # --- FIXED CONTAINER 1: Skills Matched (100% HTML in one go) ---
                    with col_a:
                        if matched:
                            chips = "".join([f'<span class="chip chip-green">{s}</span>' for s in matched])
                        else:
                            chips = "<span style='color:#94a3b8;'>No matched skills found.</span>"
                        
                        matched_html = f"""
                        <div class="result-box">
                            <h4>✅ Skills Matched</h4>
                            <div>
                                {chips}
                            </div>
                        </div>
                        """
                        st.markdown(matched_html, unsafe_allow_html=True)

                    # --- FIXED CONTAINER 2: Identified Gaps (100% HTML in one go) ---
                    with col_b:
                        if missing:
                            chips = "".join([f'<span class="chip chip-red">{s}</span>' for s in missing])
                        else:
                            chips = "<span style='color:#22c55e; font-weight:600;'>No gaps found! Perfect match.</span>"
                        
                        gaps_html = f"""
                        <div class="result-box">
                            <h4>❌ Identified Gaps</h4>
                            <div>
                                {chips}
                            </div>
                        </div>
                        """
                        st.markdown(gaps_html, unsafe_allow_html=True)

                    # Row 2: Score + Advice
                    st.markdown("---")
                    st.markdown("### 💡 Personalized Advice")

                    col_score, col_advice = st.columns([1, 3])

                    with col_score:
                        st.metric(label="Match Score", value=f"{match_percentage}%")
                        st.progress(match_percentage / 100)

                    with col_advice:
                        if mode_str == "Candidate":
                            if match_percentage >= 80:
                                advice = "🌟 Excellent match! You are highly qualified for this role. Proceed with confidence."
                            elif match_percentage >= 50:
                                advice = f"📈 Good foundation. Consider upskilling in: {', '.join(missing[:3]) if missing else 'None'}."
                            elif match_percentage >= 30:
                                advice = f"⚠️ Significant gaps. Focus on learning: {', '.join(missing[:5])}. Take relevant courses."
                            else:
                                advice = f"🔄 Career pivot needed. Missing: {', '.join(missing[:5])}. Explore entry-level roles."
                        else:
                            if match_percentage >= 70:
                                advice = "📞 Shortlist: Strong match. Proceed to interview."
                            elif match_percentage >= 40:
                                advice = f"🧐 Review: Check missing skills: {', '.join(missing[:3])}. Verify necessity."
                            else:
                                advice = f"❌ Reject: Not a fit. Missing: {', '.join(missing[:5])}."

                        st.markdown(f'<div class="advice-box"><strong>🤖 AI Insight:</strong> {advice}</div>', unsafe_allow_html=True)

                else:
                    st.warning("Could not extract meaningful text from the PDFs.")
            else:
                st.error("Failed to extract text. Please ensure the PDFs are text-based.")
    else:
        st.warning("⚠️ Please upload both a CV and a Job Description.")

# ------------------ FOOTER (Now with Beautiful Teal Gradient) ------------------
st.markdown("""
<div class="footer">
    <div class="footer-left">
        🔍 AI SkillBridge &mdash; Intelligent CV Screening
    </div>
    <div>
        <a href="#">About Us</a>
        <a href="#">Subscriptions</a>
        <a href="#">Contact</a>
        <a href="#">Privacy Policy</a>
        <a href="#">&copy; 2026</a>
    </div>
</div>
""", unsafe_allow_html=True)
