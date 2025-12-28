import streamlit as st
from src.preprocessing import clean_text
from src.features import compute_similarity
from src.model import map_similarity_to_marks

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Automated Answer Evaluation System",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
body {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Main container */
.main {
    background-color: white;
    padding: 2.5rem;
    border-radius: 16px;
    box-shadow: 0px 15px 40px rgba(0,0,0,0.15);
}

/* Headings */
h1 {
    background: linear-gradient(90deg, #4f46e5, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

h3 {
    color: #374151;
}

/* Section cards */
.card {
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
    padding: 1.5rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    border-left: 6px solid #6366f1;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg, #6366f1, #06b6d4);
    color: white;
    font-size: 1rem;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.75rem;
    border: none;
}

.stButton button:hover {
    background: linear-gradient(90deg, #4f46e5, #0891b2);
}

/* Result box */
.result {
    background: linear-gradient(135deg, #ecfeff, #eef2ff);
    padding: 1.2rem;
    border-radius: 14px;
    border-left: 6px solid #22d3ee;
    font-size: 1.05rem;
    color: #111827;   /* <-- FIX: dark readable text */
}

/* Footer */
.footer {
    text-align: center;
    font-size: 0.85rem;
    color: #6b7280;
    margin-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- APP START ----------------
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("Automated Answer Evaluation System")
st.write(
    "✨ **AI-powered academic evaluation portal** that automatically assesses "
    "descriptive answers using Natural Language Processing."
)

# ---------------- INPUT CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📘 Input Section")

question = st.text_area("📌 Question")
model_answer = st.text_area("✅ Model Answer")
student_answer = st.text_area("✍️ Student Answer")
total_marks = st.slider("🎯 Total Marks", 1, 20, 5)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- EVALUATION ----------------
if st.button("🚀 Evaluate Answer", use_container_width=True):
    if model_answer.strip() == "" or student_answer.strip() == "":
        st.warning("Please fill in all required fields.")
    else:
        model_clean = clean_text(model_answer)
        student_clean = clean_text(student_answer)

        similarity = compute_similarity(
            [model_clean],
            [student_clean]
        )[0]

        predicted_marks = map_similarity_to_marks(
            similarity,
            total_marks
        )

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Evaluation Result")

        st.markdown(
            f"""
            <div class="result">
                🔗 <b>Similarity Score:</b> {similarity:.2f}<br>
                🏆 <b>Predicted Marks:</b> {predicted_marks:.2f} / {total_marks}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="footer">
        © 2025 | Automated Answer Evaluation System <br>
        NLP & Machine Learning Academic Project
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)
