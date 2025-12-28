from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(model_answers, student_answers):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(model_answers + student_answers)

    model_vecs = tfidf[:len(model_answers)]
    student_vecs = tfidf[len(model_answers):]

    return cosine_similarity(student_vecs, model_vecs).diagonal()