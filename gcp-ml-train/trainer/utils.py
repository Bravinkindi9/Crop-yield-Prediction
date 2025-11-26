def log_message(message):
    print(f"[INFO] {message}")

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "r2": r2}

def save_model(model, filepath):
    import joblib
    joblib.dump(model, filepath)

def load_model(filepath):
    import joblib
    return joblib.load(filepath)