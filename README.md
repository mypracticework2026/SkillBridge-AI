How It Works 
Workflow
Upload

User uploads a Candidate CV (PDF) and a Job Description (PDF).

Extraction & Cleaning

pdfplumber extracts raw text from both PDFs.

The clean_text() function cleans the text: lowercasing, removing numbers/punctuation, filtering out stopwords (the, is, at), and lemmatizing words (e.g., running → run).

Vectorization

A pre‑saved TfidfVectorizer model (aicvscreening_model.pkl) converts the cleaned text into numerical vectors (TF‑IDF features).

This model was fitted on a batch of sample CVs and JDs, so it knows which words are rare/important.

Similarity Scoring

cosine_similarity calculates the angle between the CV vector and the JD vector.

A score close to 1 (100%) means the CV and JD share a high proportion of important keywords.

Skill Extraction

The app extracts the Top 15 keywords from the JD.

It checks which of these keywords appear in the CV.

Matched → displayed in green.

Missing → displayed in red as "Identified Gaps".

Personalized Advice

Based on the selected mode (Candidate or Recruiter) and the match score, the app generates tailored recommendations:

Candidate: "Significant gaps. Focus on learning: sets, aiml..."

Recruiter: "Reject: Not a fit. Missing: sets, aiml..."

Dashboard Output

The results are displayed in a clean, professional dashboard with dotted containers, a progress bar, and human‑readable advice.
