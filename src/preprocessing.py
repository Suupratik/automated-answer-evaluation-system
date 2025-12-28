import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r"\s+", " ", text).strip()

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]

    return " ".join(tokens)


def preprocess_dataframe(df):
    df = df.copy()

    df["model_answer_clean"] = df["model_answer"].apply(clean_text)
    df["student_answer_clean"] = df["student_answer"].apply(clean_text)

    df["teacher_marks"] = pd.to_numeric(df["teacher_marks"], errors="coerce")
    df["total_marks"] = pd.to_numeric(df["total_marks"], errors="coerce")

    df = df.dropna(subset=["teacher_marks", "total_marks"])

    return df
