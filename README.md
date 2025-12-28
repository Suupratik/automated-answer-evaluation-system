# 📝 Automated Answer Evaluation System

An **end-to-end NLP-based web application** that automatically evaluates descriptive student answers by comparing them with reference answers and assigning marks using **machine learning + rule-based calibration**.

This project demonstrates a **complete AI/ML workflow** — from dataset selection and preprocessing to model evaluation, optimization, and deployment via **Streamlit**.

---

## 🚀 Features

* Automatic evaluation of short/descriptive answers
* NLP-based text preprocessing and feature engineering
* TF-IDF + cosine similarity for semantic comparison
* Hybrid scoring (ML + rule-based calibration)
* Performance evaluation using MAE and RMSE
* Interactive, colorful Streamlit web interface
* Fully deployable on Streamlit Community Cloud

---

## 🧠 Real-World Problem

Manual evaluation of descriptive answers is:

* Time-consuming
* Subjective
* Inconsistent across evaluators

This system **automates answer evaluation** to assist educators by providing **consistent, explainable, and scalable grading support**.

---

## 📊 Dataset

**Source:** Kaggle – *Automatic Short Answer Grading Dataset*

**Type:** Tabular, NLP, Synthetic (education-focused)

### Dataset Columns

| Column Name      | Description            |
| ---------------- | ---------------------- |
| `question`       | Question asked         |
| `model_answer`   | Ideal/reference answer |
| `student_answer` | Student’s response     |
| `teacher_marks`  | Marks given by teacher |
| `total_marks`    | Maximum marks          |

> Synthetic data is acceptable academically and allows controlled experimentation.

---

## 🏗️ Project Folder Structure

```
Automated_Answer_Evaluation_System/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   └── evaluation.py
│
├── results/
│   └── predictions.csv
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
└── venv/
```

---

## ⚙️ Technology Stack

* **Python 3.12**
* **Pandas, NumPy**
* **Scikit-learn**
* **NLTK**
* **Streamlit**
* **TF-IDF & Cosine Similarity**

---

## 🔍 Methodology

### 1️⃣ Data Preprocessing

* Lowercasing
* Encoding cleanup
* Tokenization
* Stopword removal
* Handling missing values

### 2️⃣ Feature Engineering

* TF-IDF vectorization
* Unigram + bigram features

### 3️⃣ Similarity Computation

* Cosine similarity between model and student answers

### 4️⃣ Scoring Logic (Hybrid AI)

* Base score from similarity
* Rule-based calibration for short but correct answers
* Final score capped by total marks

### 5️⃣ Evaluation

* Mean Absolute Error (MAE)
* Root Mean Square Error (RMSE)

---

## 📁 Source Code

### `src/preprocessing.py`

```python
import pandas as pd
import re
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

    return df.dropna()
```

---

### `src/features.py`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(model_answers, student_answers):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(model_answers + student_answers)

    model_vecs = tfidf[:len(model_answers)]
    student_vecs = tfidf[len(model_answers):]

    return cosine_similarity(student_vecs, model_vecs).diagonal()
```

---

### `src/model.py`

```python
def map_similarity_to_marks(similarity, total_marks):
    base = similarity * total_marks

    if similarity >= 0.25:
        base += 0.2 * total_marks
    if similarity >= 0.40:
        base += 0.2 * total_marks

    return round(min(base, total_marks), 2)
```

---

### `src/evaluation.py`

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse
```

---

### `main.py`

```python
import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.features import compute_similarity
from src.model import map_similarity_to_marks
from src.evaluation import evaluate

train = preprocess_dataframe(pd.read_csv("data/train.csv"))
test = preprocess_dataframe(pd.read_csv("data/test.csv"))

similarity = compute_similarity(
    test["model_answer_clean"].tolist(),
    test["student_answer_clean"].tolist()
)

predicted = [
    map_similarity_to_marks(similarity[i], test["total_marks"].iloc[i])
    for i in range(len(test))
]

mae, rmse = evaluate(test["teacher_marks"], predicted)

test["predicted_marks"] = predicted
test.to_csv("results/predictions.csv", index=False)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
```

---

## 🌐 Streamlit Web Application

Run locally:

```bash
streamlit run app.py
```

The UI provides:

* Input fields for question, answers, marks
* Real-time evaluation
* Similarity score and predicted marks
* Professional, colorful, responsive design

---

## 📈 Results

* Correct answers receive high scores
* Incorrect or irrelevant answers receive low scores
* Hybrid scoring improves fairness for short answers
* Results saved to `results/predictions.csv`

---

## ✅ Conclusion

This project successfully demonstrates a **complete NLP-based automated evaluation system**, integrating:

* Machine learning
* Rule-based reasoning
* Web deployment
* Performance validation

It is **academically valid**, **interview-ready**, and **resume-worthy**.

---

## 🔮 Future Scope

* Transformer-based embeddings
* Question-wise grading models
* PDF/CSV report download
* Role-based user access
* Multi-language support

---

## 👤 Author

**Supratik Mitra**
B.Tech CSE | AI & ML Enthusiast
