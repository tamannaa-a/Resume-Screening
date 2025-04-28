# installing required libraries
!pip install pdfplumber nltk scikit-learn

# importing necessary modules
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

# download NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# upload Resume pdf
from google.colab import files
uploaded = files.upload()

# To extract text from pdf
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

resume_text = extract_text_from_pdf("")

import nltk

# re-download of punkt and related corpora
nltk.download('punkt', force=True)
nltk.download('stopwords', force=True)
nltk.download('wordnet', force=True)
nltk.download('omw-1.4', force=True)

import shutil
shutil.rmtree('/root/nltk_data', ignore_errors=True)

import nltk

nltk.download('punkt')        # Tokenizer
nltk.download('stopwords')    # Stopwords
nltk.download('wordnet')      # Lemmatizer support
nltk.download('omw-1.4')      # Lemma translation data

import shutil
shutil.rmtree('/root/nltk_data', ignore_errors=True)

import os
import nltk

# Create a new clean directory for nltk data
nltk_data_path = "/content/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)

# Set this as the nltk data directory
nltk.data.path = [nltk_data_path]

# Download everything freshly to the new directory
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

import shutil
shutil.rmtree('/root/nltk_data', ignore_errors=True)

import os
import nltk

# New custom download path
nltk_data_path = "/content/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)

# Force NLTK to look here
nltk.data.path.clear()
nltk.data.path.append(nltk_data_path)

# Download all required packages here
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('omw-1.4', download_dir=nltk_data_path)

!rm -rf /root/nltk_data

import nltk
import os

nltk_path = "/content/nltk_data"
os.makedirs(nltk_path, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(nltk_path)
nltk.download('punkt', download_dir=nltk_path)
nltk.download('stopwords', download_dir=nltk_path)
nltk.download('wordnet', download_dir=nltk_path)
nltk.download('omw-1.4', download_dir=nltk_path)

!pip install -U spacy
!python -m spacy download en_core_web_sm

# Preprocessing with SpaCy
import spacy
nlp = spacy.load("en_core_web_sm")
def spacy_preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)
sample_text = "Testing the preprocessing using SpaCy instead of NLTK!"
print(spacy_preprocess(sample_text))

import spacy

nlp = spacy.load("en_core_web_sm")


def spacy_preprocess(text):
    doc = nlp(text.lower())  # Convert text to lowercase
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

sample_text = "Natural Language Processing with SpaCy is really efficient and cool!"
processed_text = spacy_preprocess(sample_text)

print("Original Text:\n", sample_text)
print("\nProcessed Text:\n", processed_text)

from sklearn.feature_extraction.text import TfidfVectorizer

# Sample job descriptions
job_descriptions = [
    "Looking for a software engineer with experience in Python, Django, and REST APIs.",
    "Seeking data analyst skilled in Excel, SQL, and Power BI for data visualization.",
    "Hiring a machine learning engineer proficient in Python, Scikit-learn, and TensorFlow."
]

processed_jobs = [spacy_preprocess(desc) for desc in job_descriptions]

# Combine with resume
all_texts = [processed_text] + processed_jobs  # processed_text is the one from your resume

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Show feature names and matrix shape
print("TF-IDF Feature Names:\n", vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

resume_text = """
Experienced Software Engineer with strong Python skills, knowledge of REST APIs,
experience in Django, SQL, and cloud platforms. Familiar with TensorFlow and scikit-learn.
Proficient in data analysis and machine learning. Seeking a backend engineering role.
"""
import pandas as pd
job_descriptions = {
    "Software Engineer": "Looking for a skilled Python developer experienced in Django, REST APIs, and SQL.",
    "Data Analyst": "Seeking a data analyst with experience in Excel, Power BI, and data visualization.",
    "ML Engineer": "We require someone with TensorFlow, scikit-learn, and machine learning experience.",
    "Full Stack Developer": "Experience with React, Node.js, and backend systems including SQL and Django."
}
job_df = pd.DataFrame(job_descriptions.items(), columns=["Job_Title", "Job_Description"])

resume_text = """
Experienced Software Engineer with strong Python skills, knowledge of REST APIs,
experience in Django, SQL, and cloud platforms. Familiar with TensorFlow and scikit-learn.
Proficient in data analysis and machine learning. Seeking a backend engineering role.
"""
print("Resume:\n", resume_text)

print("\nJob Descriptions:")
for i, row in job_df.iterrows():
    print(f"\nJob {i+1} ({row['Job_Title']}):\n{row['Job_Description']}")

!pip install -U spacy
!python -m spacy download en_core_web_sm
import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

import spacy
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

import pandas as pd

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

job_df = pd.DataFrame(job_descriptions.items(), columns=["Job_Title", "Job_Description"])

resume_processed = preprocess_text(resume_text)
job_df["Processed_Job_Description"] = job_df["Job_Description"].apply(preprocess_text)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine resume and job descriptions into one list for vectorization
documents = [resume_processed] + job_df["Processed_Job_Description"].tolist()

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Show feature names and shape
print("TF-IDF Feature Names:\n", vectorizer.get_feature_names_out())
print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)

# Compute cosine similarity between the resume and each job description
resume_vector = tfidf_matrix[0]
job_vectors = tfidf_matrix[1:]

similarities = cosine_similarity(resume_vector, job_vectors)[0]

# Show similarity with each job
for i, score in enumerate(similarities):
    print(f"Similarity with Job {i+1} ({job_df['Job_Title'][i]}): {score:.4f}")

# Get the best match
best_match_index = similarities.argmax()
best_match_title = job_df["Job_Title"][best_match_index]
print(f"\n Best matched job: {best_match_title} with similarity {similarities[best_match_index]:.4f}")

# Sorted list of job matches
sorted_jobs = sorted(zip(job_df["Job_Title"], similarities), key=lambda x: x[1], reverse=True)

print("\n Job Ranking:")
for i, (title, score) in enumerate(sorted_jobs, start=1):
    print(f"{i}. {title} - Similarity: {score:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(job_df["Job_Title"], similarities, color='skyblue')
plt.title("Resume Similarity with Job Descriptions")
plt.ylabel("Cosine Similarity")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
