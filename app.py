import streamlit as st
import PyPDF2
from similarity import calculate_similarity

st.title("📄 AI Resume Analyzer")
st.write("Match your resume with job description using NLP")

# Upload PDF
uploaded_file = st.file_uploader("Upload Resume (PDF only)", type=["pdf"])

resume_text = ""

if uploaded_file is not None:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    for page in pdf_reader.pages:
        resume_text += page.extract_text()

job_text = st.text_area("Enter Job Description")

if st.button("Analyze Match"):

    if resume_text and job_text:
        score, matched, missing = calculate_similarity(resume_text, job_text)

        st.subheader("🔎 Results")
        st.write(f"Match Percentage: {score}%")

        if score > 60:
            st.success("Good Match ✅")
        else:
            st.error("Low Match ❌")

        st.write("Matched Skills:", matched)
        st.write("Missing Skills:", missing)

    else:
        st.warning("Please upload resume and enter job description.")