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

# ------------------ GLOBAL CSS FOR LANDING PAGE ------------------
st.markdown("""
<style>
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Reset body */
    .stApp {
        background: #F9F7F2;
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
    }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
        max-width: 1200px;
    }

    /* ----- NAVBAR (Sticky) ----- */
    .navbar {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(12px);
        padding: 0.8rem 2rem;
        border-bottom: 1px solid #E5DCC5;
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 1000;
        margin-bottom: 2rem;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 20px rgba(44,42,36,0.04);
    }
    .navbar-brand {
        font-size: 1.6rem;
        font-weight: 800;
        color: #1F4D3E;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .navbar-brand span {
        background: linear-gradient(135deg, #1F4D3E, #2E7D5B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .nav-links {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    .nav-links a {
        text-decoration: none;
        color: #2C2A24;
        font-weight: 500;
        transition: color 0.2s;
    }
    .nav-links a:hover {
        color: #1F4D3E;
    }
    .nav-cta {
        background: #1F4D3E;
        color: #F7F3EA !important;
        padding: 8px 20px;
        border-radius: 40px;
        font-weight: 600 !important;
    }
    .nav-cta:hover {
        background: #2E7D5B !important;
        color: white !important;
    }

    /* ----- HERO SECTION ----- */
    .hero {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 2rem 1rem 3rem 1rem;
        gap: 2rem;
        flex-wrap: wrap;
    }
    .hero-left {
        flex: 1;
        min-width: 280px;
    }
    .hero-left h1 {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1F4D3E;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .hero-left h1 .highlight {
        background: linear-gradient(135deg, #1F4D3E, #2E7D5B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-left p {
        font-size: 1.2rem;
        color: #5A5448;
        max-width: 500px;
        margin-bottom: 2rem;
    }
    .hero-right {
        flex: 0 0 200px;
        text-align: center;
        background: #E1EDE6;
        padding: 2rem;
        border-radius: 30px;
        border: 2px dashed #2E7D5B;
        animation: float 4s ease-in-out infinite;
    }
    .hero-right .big-robot {
        font-size: 6rem;
        display: block;
        line-height: 1;
    }
    .hero-right .sub {
        color: #1F4D3E;
        font-weight: 600;
        margin-top: 10px;
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
        100% { transform: translateY(0px); }
    }

    /* ----- FEATURES SECTION ----- */
    .features {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
        padding: 1rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(44,42,36,0.04);
        border: 1px solid #E5DCC5;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(31, 77, 62, 0.08);
    }
    .feature-card .icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .feature-card h3 {
        color: #1F4D3E;
        margin: 0.5rem 0;
    }
    .feature-card p {
        color: #6B6656;
        font-size: 0.9rem;
        margin: 0;
    }

    /* ----- TOOL SECTION (The actual app) ----- */
    .tool-container {
        background: white;
        border-radius: 30px;
        padding: 2.5rem 2rem;
        box-shadow: 0 20px 60px rgba(31, 77, 62, 0.08);
        border: 1px solid #E5DCC5;
        margin: 2rem 0 3rem 0;
    }
    .tool-container h2 {
        color: #1F4D3E;
        text-align: center;
        margin-top: 0;
    }

    /* Progress bars & chips (already defined in previous versions) */
    .custom-progress { height: 12px; border-radius: 10px; background: #EFE7D6; overflow: hidden; margin: 0.5rem 0 1rem 0; }
    .custom-progress-fill { height: 100%; border-radius: 10px; background: linear-gradient(90deg, #1F4D3E, #2E7D5B); width: 0%; transition: width 0.8s ease; }
    .match-score-number { font-size: 3.5rem; font-weight: 800; background: linear-gradient(135deg, #1F4D3E, #2E7D5B); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1; }
    .skill-chip-green { background: #E1EDE6; color: #1F4D3E; padding: 6px 14px; border-radius: 30px; display: inline-block; margin: 4px 6px 4px 0; font-size: 0.85rem; font-weight: 500; border: 1px solid #B9D9C7; }
    .skill-chip-red { background: #F3E4D8; color: #8A4A2D; padding: 6px 14px; border-radius: 30px; display: inline-block; margin: 4px 6px 4px 0; font-size: 0.85rem; font-weight: 500; border: 1px solid #E3C2A8; }
    .upload-card { background: #F9F7F2; padding: 1.5rem; border-radius: 16px; border: 1px solid #E5DCC5; height: 100%; }
    .stButton button { background: linear-gradient(135deg, #1F4D3E 0%, #2E7D5B 100%); color: #F7F3EA !important; font-weight: 600 !important; border: none !important; border-radius: 12px !important; padding: 0.6rem 2rem !important; box-shadow: 0 4px 14px rgba(31, 77, 62, 0.35); width: 100%; }
    .stButton button:hover { transform: scale(1.02); box-shadow: 0 8px 25px rgba(31, 77, 62, 0.5); }
    .stRadio > div { display: flex; gap: 20px; background: #F9F7F2; padding: 10px 20px; border-radius: 40px; border: 1px solid #E5DCC5; }
    .stRadio label[data-baseweb="radio"]:has(input:checked) { background: linear-gradient(135deg, #1F4D3E, #2E7D5B); color: #F7F3EA !important; border-radius: 30px; padding: 8px 20px; }

    /* Footer */
    .footer {
        text-align: center;
        color: #6B6656;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid #E5DCC5;
        margin-top: 2rem;
        font-size: 0.8rem;
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

# ------------------ LANDING PAGE HTML (Navbar, Hero, Features) ------------------
st.markdown("""
<!-- NAVBAR -->
<div class="navbar">
    <div class="navbar-brand">🔍 <span>AI SkillBridge</span></div>
    <div class="nav-links">
        <a href="#">Features</a>
        <a href="#">Pricing</a>
        <a href="#" class="nav-cta">Get Started</a>
    </div>
</div>

<!-- HERO -->
<div class="hero">
    <div class="hero-left">
        <h1>Find Your <span class="highlight">Perfect Match</span> with AI.</h1>
        <p>Upload your CV and Job Description. Get a detailed skill gap analysis and personalized advice — whether you're hiring or applying.</p>
    </div>
    <div class="hero-right">
        <div class="big-robot">🤖</div>
        <div class="sub">🔍 AI Scanning</div>
        <div style="font-size:0.8rem; color:#5A5448;">Real-time analysis</div>
    </div>
</div>

<!-- FEATURES -->
<div class="features">
    <div class="feature-card">
        <div class="icon">📄</div>
        <h3>Instant Parsing</h3>
        <p>Reads your PDFs and extracts key skills in seconds.</p>
    </div>
    <div class="feature-card">
        <div class="icon">🎯</div>
        <h3>Skill Gap Analysis</h3>
        <p>Matches your CV against the JD and highlights exactly what's missing.</p>
    </div>
    <div class="feature-card">
        <div class="icon">👤</div>
        <h3>Dual Perspectives</h3>
        <p>Switch between Candidate (upskill) and Recruiter (hire) views.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ TOOL SECTION (The actual app) ------------------
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

                    # --- RESULTS (First Row: Score + Skills) ---
                    st.markdown("---")
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

                    st.markdown("---")

                    # --- RESULTS (Second Row: AI Robot + BULLETPROOF HTML ADVICE) ---
                    advice_col1, advice_col2 = st.columns([1, 3], gap="medium")

                    with advice_col1:
                        st.markdown("""
                        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; background:rgba(31,77,62,0.03); border-radius:20px; padding:15px; border:1px dashed #2E7D5B; height:100%; min-height:180px;">
                            <div style="font-size:4rem; animation:float 3s ease-in-out infinite;">🤖</div>
                            <div style="font-size:1.5rem; animation:resume-sway 3s ease-in-out infinite; margin-top:-10px;">📄</div>
                            <div style="background:#E1EDE6; padding:4px 14px; border-radius:30px; font-weight:600; color:#1F4D3E; font-size:0.75rem; margin-top:5px;">🔍 AI Scanning</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with advice_col2:
                        if missing:
                            # Build the list items as pure HTML string
                            list_items = ""
                            for idx, skill in enumerate(missing[:7], start=1):
                                list_items += f"""
                                <li style="display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid #F0EBE0;">
                                    <span style="background: #1F4D3E; color: #F7F3EA; font-weight: 700; font-size: 0.75rem; width: 24px; height: 24px; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; flex-shrink: 0;">{idx}</span>
                                    <span style="flex: 1; color: #2C2A24; font-size: 0.9rem;">Gap detected in <strong>'{skill}'</strong></span>
                                    <span style="background: #F3E4D8; color: #8A4A2D; padding: 2px 12px; border-radius: 30px; font-weight: 500; font-size: 0.75rem; white-space: nowrap;">Priority {idx}</span>
                                </li>
                                """

                            if mode_str == "Candidate":
                                title = "📌 Personalized Upskilling Plan"
                                action_msg = "📚 Consider taking relevant courses or building projects to bridge these gaps."
                            else:
                                title = "📌 Hiring Recommendation (Skill Gaps)"
                                action_msg = "🧐 Verify if these missing skills are strictly required for the role."

                            # Build the FULL HTML document for the iframe
                            advice_html = f"""
                            <div style="background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.8rem; border-left: 6px solid #2E7D5B; box-shadow: 0 8px 24px rgba(31, 77, 62, 0.08); height: 100%; min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #1F4D3E; font-weight: 700; margin-top: 0; margin-bottom: 10px; font-size: 1.1rem;">{title}</h4>
                                <ul style="list-style: none; padding: 0; margin: 0;">
                                    {list_items}
                                </ul>
                                <div style="margin-top: 12px; font-size: 0.9rem; color: #1F4D3E; background: #E1EDE6; padding: 8px 16px; border-radius: 30px; display: inline-block; font-weight: 500; align-self: flex-start;">💡 {action_msg}</div>
                            </div>
                            """
                            # RENDER USING st.components.v1.html (100% bulletproof, no Markdown interference!)
                            st.components.v1.html(advice_html, height=280)
                        else:
                            perfect_html = """
                            <div style="background: #FFFFFF; border-radius: 16px; padding: 1.2rem 1.8rem; border-left: 6px solid #4CAF50; box-shadow: 0 8px 24px rgba(31, 77, 62, 0.08); height: 100%; min-height: 180px; display: flex; flex-direction: column; justify-content: center;">
                                <h4 style="color: #1F4D3E; font-weight: 700; margin-top: 0; font-size: 1.1rem;">🎯 Verdict</h4>
                                <p style="font-size:1.1rem; font-weight:500; color:#1F4D3E; margin:0;">🌟 Perfect Match! You cover all the key requirements. Proceed with confidence!</p>
                            </div>
                            """
                            st.components.v1.html(perfect_html, height=180)
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
