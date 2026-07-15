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
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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
        st.error("🚨 Model file 'aicvscreening_model.pkl' not found! Please ensure it's in the same directory.")
        st.stop()

model = load_model()
vectorizer = model['vectorizer']
# We don't actually need the old cv_texts/jd_texts for the direct comparison, 
# but we need the feature names from the vectorizer.
feature_names = vectorizer.get_feature_names_out()

# ------------------ TEXT CLEANING FUNCTION (IDENTICAL TO COLAB) ------------------
def wash_text(raw_text):
    if not raw_text:
        return ""
    # 1. Lowercase
    text = raw_text.lower()
    # 2. Keep ONLY letters and spaces (removes numbers, punctuation, symbols)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 3. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 4. Split, remove stopwords, keep words longer than 2 chars
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words and len(word) > 2]
    # 5. Join back
    return " ".join(cleaned_words)

# ------------------ EXTRACT TEXT FROM PDF ------------------
def extract_text_from_pdf(uploaded_file):
    try:
        with pdfplumber.open(io.BytesIO(uploaded_file.getbuffer())) as pdf:
            full_text = ""
            for page in pdf.pages:
                full_text += page.extract_text() or ""
        return full_text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# ------------------ MAIN ANALYSIS FUNCTION ------------------
def analyze_cv_jd(cv_text, jd_text, mode):
    # 1. Clean texts
    cv_cleaned = wash_text(cv_text)
    jd_cleaned = wash_text(jd_text)
    
    if not cv_cleaned or not jd_cleaned:
        st.warning("Could not extract meaningful text from one of the files. Please check the PDFs.")
        return
    
    # 2. Transform using the LOADED vectorizer (DO NOT FIT AGAIN)
    cv_vector = vectorizer.transform([cv_cleaned])
    jd_vector = vectorizer.transform([jd_cleaned])
    
    # 3. Calculate Similarity
    score = cosine_similarity(cv_vector, jd_vector)[0][0]
    match_percentage = round(score * 100, 2)
    
    # 4. Extract Top 15 Skills from the JD
    jd_array = jd_vector.toarray().flatten()
    word_score_pairs = sorted(zip(feature_names, jd_array), key=lambda x: x[1], reverse=True)
    required_skills = [word for word, w_score in word_score_pairs if w_score > 0][:15]
    
    # 5. Check CV against these skills
    cv_words = set(cv_cleaned.split())
    matched = [skill for skill in required_skills if skill in cv_words]
    missing = [skill for skill in required_skills if skill not in cv_words]
    
    # 6. Generate Advice
    advice = ""
    action = ""
    decision = ""
    risk = ""
    
    if mode == "Candidate":
        if match_percentage >= 80:
            advice = "🌟 EXCELLENT MATCH! You are highly qualified."
            action = "🚀 Proceed with confidence. Highlight these matched skills."
        elif match_percentage >= 50:
            advice = f"📈 GOOD FOUNDATION. Upskill in: {', '.join(missing[:3])}."
            action = "💡 Consider a short certification."
        elif match_percentage >= 30:
            advice = f"⚠️ SIGNIFICANT GAPS. Focus on: {', '.join(missing[:5])}."
            action = "📚 Take relevant courses."
        else:
            advice = f"🔄 CAREER PIVOT NEEDED. Missing: {', '.join(missing[:5])}."
            action = "🧭 Explore junior or adjacent roles."
    else:  # Recruiter
        if match_percentage >= 70:
            decision = "📞 SHORTLIST: Strong match."
            risk = "Low risk. Proceed to interview."
        elif match_percentage >= 40:
            decision = f"🧐 REVIEW: Check missing: {', '.join(missing[:3])}."
            risk = "Medium risk."
        elif match_percentage >= 20:
            decision = f"🗄️ HOLD: Low match. Missing: {', '.join(missing[:5])}."
            risk = "High risk."
        else:
            decision = "❌ REJECT: Not suitable."
            risk = "Proceed with other candidates."

    # 7. Display Results
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### 🏆 Match Score")
        # Progress bar for score
        st.progress(match_percentage / 100)
        st.markdown(f"## <span style='color:#4CAF50;'>{match_percentage}%</span>", unsafe_allow_html=True)
        
        if mode == "Candidate":
            st.info(f"**💡 Advice:** {advice}")
            st.success(f"**🎯 Action:** {action}")
        else:
            st.info(f"**⚖️ Decision:** {decision}")
            st.warning(f"**⚠️ Risk:** {risk}")

    with col2:
        st.markdown(f"### 📊 Skills Breakdown")
        
        # Matched Skills
        matched_str = ', '.join(matched) if matched else "None"
        st.markdown(f"**✅ Matched ({len(matched)})**")
        st.markdown(f"<span style='background-color:#e8f5e9; padding:5px; border-radius:5px;'>{matched_str}</span>", unsafe_allow_html=True)
        
        # Missing Skills
        missing_str = ', '.join(missing) if missing else "None"
        st.markdown(f"**❌ Missing ({len(missing)})**")
        st.markdown(f"<span style='background-color:#ffebee; padding:5px; border-radius:5px;'>{missing_str}</span>", unsafe_allow_html=True)

# ------------------ UI LAYOUT ------------------

# ---- TOP HEADER (Brand Left, Hero Center) ----
# Define columns: col1 (left, small), col2 (center, massive), col3 (right, small)
c1, c2, c3 = st.columns([1, 4, 1])
with c1:
    st.markdown("## 🔗 AI SkillBridge")
with c2:
    st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🤖 AI CV Screening</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-top: -20px;'>Intelligent Resume vs Job Description Matcher</p>", unsafe_allow_html=True)
with c3:
    st.write("")  # Empty spacer

st.divider()

# ---- UPLOAD SECTION ----
upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    st.subheader("📄 Upload Candidate CV")
    cv_file = st.file_uploader("Choose a PDF", type=['pdf'], key="cv_upload")
    if cv_file:
        st.success(f"✅ {cv_file.name} uploaded")

with upload_col2:
    st.subheader("📌 Upload Job Description")
    jd_file = st.file_uploader("Choose a PDF", type=['pdf'], key="jd_upload")
    if jd_file:
        st.success(f"✅ {jd_file.name} uploaded")

# ---- MODE SELECTOR ----
mode = st.radio(
    "👤 Select View",
    ["🎓 Candidate Mode (Upskilling Advice)", "💼 Recruiter Mode (Hiring Decision)"],
    horizontal=True,
    index=0
)

# ---- ANALYZE BUTTON ----
if st.button("🚀 Analyze Match", type="primary", use_container_width=True):
    if cv_file and jd_file:
        with st.spinner("🔍 Reading and analyzing..."):
            cv_text = extract_text_from_pdf(cv_file)
            jd_text = extract_text_from_pdf(jd_file)
            
            if cv_text and jd_text:
                # Determine mode string for the function
                mode_str = "Candidate" if "Candidate" in mode else "Recruiter"
                analyze_cv_jd(cv_text, jd_text, mode_str)
            else:
                st.error("Failed to extract text from one or both PDFs. Please ensure they are text-based (not scanned images).")
    else:
        st.warning("⚠️ Please upload both a CV and a Job Description.")

# ---- FOOTER ----
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8em;'>Built with ❤️ using Streamlit, Scikit-Learn, and PDFPlumber</p>",
    unsafe_allow_html=True
)
