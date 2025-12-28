import pandas as pd
from src.preprocessing import preprocess_dataframe
from src.features import compute_similarity
from src.model import map_similarity_to_marks
from src.evaluation import evaluate

# Load data
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Preprocess
train_df = preprocess_dataframe(train_df)
test_df = preprocess_dataframe(test_df)

# Similarity computation
similarity_scores = compute_similarity(
    test_df["model_answer_clean"].tolist(),
    test_df["student_answer_clean"].tolist()
)

# Predict marks
predicted_marks = map_similarity_to_marks(
    similarity_scores,
    test_df["total_marks"].values
)

# Evaluation
mae, rmse = evaluate(
    test_df["teacher_marks"].values,
    predicted_marks
)

# Save results
test_df["predicted_marks"] = predicted_marks
test_df["similarity_score"] = similarity_scores
test_df.to_csv("results/predictions.csv", index=False)

print("Evaluation Results")
print("------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print("\nResults saved to results/predictions.csv")
