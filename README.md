# Resume-Screening-
This project uses NLP and TF-IDF based cosine similarity to match a candidate's resume.
The system preprocesses the text using spaCy, vectorizes both resume and job descriptions using TF-IDF, and ranks job roles based on similarity scores. It helps automate and optimize the job screening process by identifying the best-fit roles quickly and accurately.

Preprocesses resume and job descriptions using **spaCy**
- Converts text data into vectors using **TF-IDF**
- Calculates **cosine similarity** between resume and each job description
- Ranks and identifies the **best-fit job role** for the candidate
- Easy to customize with new resumes and job roles

Tech Stack
- Python
- Google Colab
- SpaCy (NLP)
- Scikit-learn (TF-IDF, Cosine Similarity)
- Pandas, NumPy
