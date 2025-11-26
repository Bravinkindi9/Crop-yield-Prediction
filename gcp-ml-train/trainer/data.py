import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data (e.g., handle missing values, encode categorical variables)."""
    # Example preprocessing steps
    data.fillna(method='ffill', inplace=True)  # Forward fill for missing values
    # Add more preprocessing steps as needed
    return data

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test