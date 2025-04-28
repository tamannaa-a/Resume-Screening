import streamlit as st
import pdfplumber
import spacy
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Function to calculate cosine similarity between two texts
def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return cosine_similarity([doc1.vector], [doc2.vector])[0][0]

# Streamlit UI
st.title("Resume Screening App")

# Upload resume
uploaded_file = st.file_uploader("Upload your resume", type=["pdf"])

if uploaded_file:
    # Extract text from the uploaded resume
    resume_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text:")
    st.write(resume_text)

    # You can add your own job description to compare with the resume
    job_description = st.text_area("Enter Job Description", "Enter job description here...")

    if job_description:
        # Calculate similarity between resume and job description
        similarity_score = calculate_similarity(resume_text, job_description)
        st.subheader(f"Similarity Score: {similarity_score:.2f}")
        
        if similarity_score > 0.7:
            st.success("The resume is a good match for the job!")
        else:
            st.warning("The resume doesn't match well with the job description.")

# Adding some extra functionality for displaying data if required
st.subheader("Resume Summary")
if uploaded_file:
    st.text("A summary of your resume can be displayed here.")
