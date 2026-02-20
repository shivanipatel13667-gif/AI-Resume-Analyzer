import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Take input
resume_text = input("Enter Resume Skills: ")
job_description = input("Enter Job Description: ")

# Clean text
resume_clean = clean_text(resume_text)
job_clean = clean_text(job_description)

# Convert to list
documents = [resume_clean, job_clean]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Similarity
similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

score = similarity_score[0][0]

print("\nMatch Percentage:", round(score * 100, 2), "%")

from text_cleaner import clean_text
from similarity import calculate_similarity

# Take input
resume_text = input("Enter Resume Skills: ")
job_description = input("Enter Job Description: ")

# Clean text
resume_clean = clean_text(resume_text)
job_clean = clean_text(job_description)

# Calculate similarity
score = calculate_similarity(resume_clean, job_clean)

print("\nMatch Percentage:", round(score * 100, 2), "%")

# Match Level
if score < 0.4:
    print("Match Level: Low")
elif score < 0.7:
    print("Match Level: Moderate")
else:
    print("Match Level: Strong")

# Matched & Missing Skills
resume_words = set(resume_clean.split())
job_words = set(job_clean.split())

matched = resume_words.intersection(job_words)
missing = job_words.difference(resume_words)

print("Matched Skills:", matched)
print("Missing Skills:", missing)
