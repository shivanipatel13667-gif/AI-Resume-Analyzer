from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(resume_text, job_text):
    text = [resume_text, job_text]

    cv = CountVectorizer().fit_transform(text)
    similarity_score = cosine_similarity(cv)[0][1]

    resume_words = set(resume_text.lower().split())
    job_words = set(job_text.lower().split())

    matched = resume_words.intersection(job_words)
    missing = job_words.difference(resume_words)

    percentage = round(similarity_score * 100, 2)

    return percentage, matched, missing