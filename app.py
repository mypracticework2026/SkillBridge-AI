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

# ------------------ VIBRANT GLOBAL CSS ------------------
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    .stApp {
        background: #f0f4f8;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }

    /* ----- NAVBAR ----- */
    .navbar {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        padding: 0.8rem 2rem;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
    }
    .navbar-brand { font-size: 1.6rem; font-weight: 800; color: #0f172a; display: flex; align-items: center; gap: 8px; }
    .navbar-brand span { color: #0d9488; }
    .nav-links { display: flex; gap: 2rem; align-items: center; }
    .nav-links a { text-decoration: none; color: #334155; font-weight: 500; }
    .nav-links a:hover { color: #0d9488; }
    .nav-cta { background: #0d9488; color: white !important; padding: 8px 20px; border-radius: 40px; font-weight: 600 !important; }
    .nav-cta:hover { background: #0f766e !important; }

    /* ----- HERO ----- */
    .hero { display: flex; align-items: center; justify-content: space-between; padding: 1rem 0 2rem 0; gap: 2rem; flex-wrap: wrap; }
    .hero-left h1 { font-size: 3.2rem; font-weight: 800; color: #0f172a; line-height: 1.1; }
    .hero-left h1 .highlight { color: #0d9488; }
    .hero-left p { font-size: 1.2rem; color: #475569; max-width: 500px; }
    .hero-right { flex: 0 0 200px; text-align: center; background: linear-gradient(135deg, #ccfbf1, #f0fdfa); padding: 2rem; border-radius: 30px; border: 2px solid #99f6e4; animation: float 4s ease-in-out infinite; }
    .hero-right .big-robot { font-size: 6rem; display: block; line-height: 1; }
    .hero-right .sub { color: #0f766e; font-weight: 700; margin-top: 10px; }
    @keyframes float { 0% { transform: translateY(0px); } 50% { transform: translateY(-15px); } 100% { transform: translateY(0px); } }

    /* ----- TOOL CONTAINER ----- */
    .tool-container { background: white; border-radius: 30px; padding: 2.5rem 2rem; box-shadow: 0 20px 60px rgba(13, 148, 136, 0.08); border: 1px solid #e2e8f0; margin: 2rem 0; }
    .tool-container h2 { color: #0f172a; text-align: center; margin-top: 0; font-weight: 700; }

    .upload-card { background: #f8fafc; padding: 1.5rem; border-radius: 16px; border: 1px solid #e2e8f0; height: 100%; }
    .upload-card:hover { border-color: #0d9488; }

    /* ----- RESULT CARDS (FOR st.components.v1.html) ----- */
    .result-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem 1.2rem;
        border: 3px dotted #0d9488;
        height: 100%;
        min-height: 260px;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .result-card .title {
        font-weight: 700;
        font-size: 1.05rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #f1f5f9;
        padding-bottom: 10px;
        margin-bottom: 12px;
    }
    .result-card .badge {
        background: #0d9488;
        color: white;
        font-size: 0.6rem;
        padding: 2px 10px;
        border-radius: 30px;
    }
    .chip-green { background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 30px; display: inline-block; margin: 3px 4px 3px 0; font-size: 0.8rem; font-weight: 600; border: 1px solid #bbf7d0; }
    .chip-red { background: #fee2e2; color: #991b1b; padding: 4px 12px; border-radius: 30px; display: inline-block; margin: 3px 4px 3px 0; font-size: 0.8rem; font-weight: 600; border: 1px solid #fecaca; }
    .chip-neutral { background: #f1f5f9; color: #475569; padding: 4px 12px; border-radius: 30px; display: inline-block; margin: 3px 4px 3px 0; font-size: 0.8rem; font-weight: 500; border: 1px solid #e2e8f0; }
    .big-score { font-size: 3.5rem; font-weight: 800; color: #0d9488; line-height: 1; text-align: center; margin: 0.2rem 0; }
    .score-label { text-align: center; font-weight: 600; color: #475569; font-size: 0.8rem; }
    .progress-bar-bg { height: 8px; border-radius: 10px; background: #f1f5f9; overflow: hidden; margin: 0.3rem 0 0.5rem 0; }
    .progress-bar-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #0d9488, #14b8a6); width: 0%; }
    .stat-line { margin-top: 10px; padding-top: 10px; border-top: 1px solid #f1f5f9; font-size: 0.85rem; color: #475569; }
    .stat-line strong { color: #0f172a; }

    /* Buttons */
    .stButton button { background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%); color: white !important; font-weight: 600 !important; border: none !important; border-radius: 12px !important; padding: 0.6rem 2rem !important; box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35); width: 100%; }
    .stButton button:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(13, 148, 136, 0.5); }
    .stRadio > div { display: flex; gap: 20px; background: #f8fafc; padding: 10px 20px; border-radius: 40px; border: 1px solid #e2e8f0; }
    .stRadio label[data-baseweb="radio"]:has(input:checked) { background: #0d9488; color: white !important; border-radius: 30px; padding: 8px 20px; }
    .footer { text-align: center; color: #94a3b8; padding: 2rem 0 1rem 0; border-top: 1px solid #e2e8f0; margin-top: 2rem; font-size: 0.8rem; }

    @media (max-width: 768px) { .hero { flex-direction: column; text-align: center; } .hero-left h1 { font-size: 2.2rem; } }
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

# ------------------ NAV & HERO ------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">🔍 <span>AI SkillBridge</span></div>
    <div class="nav-links">
        <a href="#">Features</a>
        <a href="#">Pricing</a>
        <a href="#" class="nav-cta">Get Started</a>
    </div>
</div>
<div class="hero">
    <div class="hero-left">
        <h1>Find Your <span class="highlight">Perfect Match</span> with AI.</h1>
        <p>Upload your CV and Job Description. Get a detailed skill gap analysis and personalized advice.</p>
    </div>
    <div class="hero-right">
        <div class="big-robot">🤖</div>
        <div class="sub">🔍 AI Scanning</div>
        <div style="font-size:0.8rem; color:#64748b;">Real-time analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ TOOL SECTION ------------------
st.markdown('<div class="tool-container">', unsafe_allow_html=True)
st.markdown('<h2>🚀 Live Demo: Try It Now</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📄 Upload CV")
    cv_file = st.file_uploader("Choose PDF", type=['pdf'], key="cv_upload", label_visibility="collapsed")
    if cv_file: st.success(f"✅ {cv_file.name}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📌 Upload JD")
    jd_file = st.file_uploader("Choose PDF", type=['pdf'], key="jd_upload", label_visibility="collapsed")
    if jd_file: st.success(f"✅ {jd_file.name}")
    st.markdown('</div>', unsafe_allow_html=True)

mode = st.radio(
    "👤 Select View",
    ["🎓 Candidate Mode (Upskilling)", "💼 Recruiter Mode (Hiring)"],
    horizontal=True,
    index=0
)

if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    if cv_file and jd_file:
        with st.spinner("🔍 AI is scanning the documents..."):
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

                    st.markdown("---")
                    st.markdown("### 📊 Analysis Dashboard")

                    # ======================== 3-COLUMN CARDS (FIXED: 100% HTML) ========================
                    col_jd, col_cv, col_results = st.columns([1, 1, 1.2], gap="medium")

                    # ----- COLUMN 1: JOB DESCRIPTION -----
                    with col_jd:
                        # Build the complete HTML string
                        if required_skills:
                            chips = "".join([f'<span class="chip-neutral">{skill}</span>' for skill in required_skills[:10]])
                        else:
                            chips = '<span style="color: #94a3b8;">No requirements extracted.</span>'
                        
                        jd_html = f"""
                        <div class="result-card">
                            <div class="title">📌 Job Description <span class="badge">Requirements</span></div>
                            <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;">
                                {chips}
                            </div>
                            <div style="flex: 1;"></div>
                        </div>
                        """
                        st.components.v1.html(jd_html, height=300, scrolling=False)

                    # ----- COLUMN 2: CANDIDATE CV -----
                    with col_cv:
                        if matched:
                            cv_chips = "".join([f'<span class="chip-green">{skill}</span>' for skill in matched[:12]])
                        else:
                            cv_chips = '<span style="color: #94a3b8;">No skills matched.</span>'
                        
                        cv_html = f"""
                        <div class="result-card">
                            <div class="title">📄 Candidate CV <span class="badge">Your Profile</span></div>
                            <p style="margin: 0 0 6px 0; font-weight: 600; color: #0f172a; font-size: 0.85rem;">✅ Skills Detected</p>
                            <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">
                                {cv_chips}
                            </div>
                            <div class="stat-line">
                                📊 <strong>Profile Stats</strong><br>
                                Total skills analyzed: <strong>{len(cv_words)}</strong>
                            </div>
                        </div>
                        """
                        st.components.v1.html(cv_html, height=300, scrolling=False)

                    # ----- COLUMN 3: SIMILARITIES & GAPS -----
                    with col_results:
                        if matched:
                            match_chips = "".join([f'<span class="chip-green">{skill}</span>' for skill in matched[:5]])
                        else:
                            match_chips = '<span style="color: #94a3b8;">None</span>'
                        
                        if missing:
                            gap_chips = "".join([f'<span class="chip-red">{skill}</span>' for skill in missing[:5]])
                            if len(missing) > 5:
                                gap_chips += f'<br><span style="color: #94a3b8; font-size:0.8rem;">+ {len(missing) - 5} more gaps...</span>'
                        else:
                            gap_chips = '<span style="color: #22c55e; font-weight:600;">No gaps! Perfect match.</span>'
                        
                        verdict = ""
                        if mode_str == "Candidate":
                            if match_percentage >= 80: verdict = "🌟 Excellent Fit!"
                            elif match_percentage >= 50: verdict = "📈 Good Foundation"
                            else: verdict = "🔄 Gaps Detected"
                        else:
                            if match_percentage >= 70: verdict = "📞 Shortlist"
                            elif match_percentage >= 40: verdict = "🧐 Review"
                            else: verdict = "❌ Reject"
                        
                        results_html = f"""
                        <div class="result-card">
                            <div class="title">⚡ Similarities & Gaps</div>
                            
                            <p style="margin: 0 0 2px 0; font-weight: 600; color: #0f172a; font-size: 0.85rem;">✅ Matched Skills</p>
                            <div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px;">
                                {match_chips}
                            </div>
                            
                            <div style="margin: 4px 0;">
                                <div class="big-score">{match_percentage}%</div>
                                <div class="score-label">Match Score</div>
                                <div class="progress-bar-bg">
                                    <div class="progress-bar-fill" style="width: {min(match_percentage, 100)}%;"></div>
                                </div>
                            </div>
                            
                            <div style="font-weight:600; margin: 2px 0 4px 0;">{verdict}</div>
                            
                            <div style="border-top: 1px solid #f1f5f9; padding-top: 8px; margin-top: 4px;">
                                <p style="margin: 0 0 2px 0; font-weight: 600; color: #0f172a; font-size: 0.85rem;">❌ Skill Gaps</p>
                                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                    {gap_chips}
                                </div>
                            </div>
                        </div>
                        """
                        st.components.v1.html(results_html, height=340, scrolling=False)

                    # ----- ADVICE BANNER (Below the grid) -----
                    st.markdown("---")
                    if missing:
                        if mode_str == "Candidate":
                            advice_title = "📚 Personalized Upskilling Plan"
                            action_msg = "💡 Consider taking relevant courses or building projects to bridge these gaps."
                        else:
                            advice_title = "🧐 Hiring Recommendation"
                            action_msg = "💡 Verify if these missing skills are strictly required for the role."

                        adv_col1, adv_col2 = st.columns([1, 4])
                        with adv_col1:
                            st.markdown("#### 🤖")
                        with adv_col2:
                            st.info(f"**{advice_title}**  \n{action_msg}")
                    else:
                        st.balloons()
                        st.success("🎉 **Perfect Match!** This candidate covers all key requirements. Proceed with confidence!")

                else:
                    st.warning("Could not extract meaningful text.")
            else:
                st.error("Failed to extract text from PDFs.")
    else:
        st.warning("⚠️ Please upload both a CV and a Job Description.")

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown("""
<div class="footer">
    Built with ❤️ using Streamlit, Scikit-Learn, and PDFPlumber<br>
    &copy; 2026 AI SkillBridge
</div>
""", unsafe_allow_html=True)
