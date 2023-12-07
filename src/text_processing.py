from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(text):
    vectorizer = TfidfVectorizer(stop_words="english")
    text_vectorized = vectorizer.fit_transform(text)
    return text_vectorized, vectorizer

def perform_vectorization(vectorizer, texts):
    return vectorizer.transform(texts)
