from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate(true_marks, predicted_marks):
    mae = mean_absolute_error(true_marks, predicted_marks)
    rmse = np.sqrt(mean_squared_error(true_marks, predicted_marks))

    return mae, rmse
