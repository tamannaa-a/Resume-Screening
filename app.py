# Importing necessary libraries
import pdfplumber
import nltk
import string
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
import matplotlib.pyplot as plt

# Check and download SpaCy model if not found
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading the SpaCy model...")
    !python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")

# NLTK Downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Preprocessing function using SpaCy
def spacy_preprocess(text):
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

# Sample resume and job descriptions (you can replace this with actual file uploads)
resume_text = """
Experienced Software Engineer with strong Python skills, knowledge of REST APIs,
experience in Django, SQL, and cloud platforms. Familiar with TensorFlow and scikit-learn.
Proficient in data analysis and machine learning. Seeking a backend engineering role.
"""

job_descriptions = {
    "Software Engineer": "Looking for a skilled Python developer experienced in Django, REST APIs, and SQL.",
    "Data Analyst": "Seeking a data analyst with experience in Excel, Power BI, and data visualization.",
    "ML Engineer": "We require someone with TensorFlow, scikit-learn, and machine learning experience.",
    "Full Stack Developer": "Experience with React, Node.js, and backend systems including SQL and Django."
}

# Preprocess the resume and job descriptions
resume_processed = spacy_preprocess(resume_text)
job_df = pd.DataFrame(job_descriptions.items(), columns=["Job_Title", "Job_Description"])
job_df["Processed_Job_Description"] = job_df["Job_Description"].apply(spacy_preprocess)

# Combine resume and job descriptions into one list for vectorization
documents = [resume_processed] + job_df["Processed_Job_Description"].tolist()

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity between the resume and each job description
resume_vector = tfidf_matrix[0]
job_vectors = tfidf_matrix[1:]

similarities = cosine_similarity(resume_vector, job_vectors)[0]

# Output the similarity results
for i, score in enumerate(similarities):
    print(f"Similarity with Job {i+1} ({job_df['Job_Title'][i]}): {score:.4f}")

# Get the best match
best_match_index = similarities.argmax()
best_match_title = job_df["Job_Title"][best_match_index]
print(f"\n Best matched job: {best_match_title} with similarity {similarities[best_match_index]:.4f}")

# Plot the similarity results
plt.figure(figsize=(8, 5))
plt.bar(job_df["Job_Title"], similarities, color='skyblue')
plt.title("Resume Similarity with Job Descriptions")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
