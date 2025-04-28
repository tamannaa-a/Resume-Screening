# app.py

import streamlit as st
import pdfplumber
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def main():
    st.title("AI Resume Screening App")

    uploaded_file = st.file_uploader("Upload Your Resume (PDF)", type="pdf")

    if uploaded_file is not None:
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_processed = preprocess_text(resume_text)

        st.subheader("Extracted Resume Text:")
        st.write(resume_text)

        job_descriptions = {
            "Software Engineer": "Looking for a skilled Python developer experienced in Django, REST APIs, and SQL.",
            "Data Analyst": "Seeking a data analyst with experience in Excel, Power BI, and data visualization.",
            "ML Engineer": "We require someone with TensorFlow, scikit-learn, and machine learning experience.",
            "Full Stack Developer": "Experience with React, Node.js, and backend systems including SQL and Django."
        }
        job_df = pd.DataFrame(job_descriptions.items(), columns=["Job_Title", "Job_Description"])
        job_df["Processed_Job_Description"] = job_df["Job_Description"].apply(preprocess_text)

        documents = [resume_processed] + job_df["Processed_Job_Description"].tolist()

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        resume_vector = tfidf_matrix[0]
        job_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(resume_vector, job_vectors)[0]

        best_match_index = similarities.argmax()
        best_match_title = job_df["Job_Title"][best_match_index]

        st.subheader("Best Matched Job:")
        st.success(f"{best_match_title} (Similarity Score: {similarities[best_match_index]:.2f})")

        st.subheader("Job Matching Scores:")

        sorted_jobs = sorted(zip(job_df["Job_Title"], similarities), key=lambda x: x[1], reverse=True)
        for i, (title, score) in enumerate(sorted_jobs, start=1):
            st.write(f"{i}. {title} - Similarity: {score:.2f}")

        st.subheader("Similarity Chart:")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(job_df["Job_Title"], similarities, color='skyblue')
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
