def map_similarity_to_marks(similarity, total_marks):
    """
    Hybrid scoring:
    - Boost scores for semantically acceptable short answers
    - Keep TF-IDF as base
    """

    # Base score
    base_score = similarity * total_marks

    # Calibration rules
    if similarity >= 0.25:
        base_score += 0.2 * total_marks  # semantic boost

    if similarity >= 0.40:
        base_score += 0.2 * total_marks

    # Cap at total marks
    final_score = min(base_score, total_marks)

    return round(final_score, 2)
