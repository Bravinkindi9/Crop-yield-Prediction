import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Set paths
DATA_PATH = r"C:\Users\USER\Desktop\Projects_git\Crop yield dataset\dataset\yield_df.csv"
ARTIFACTS_DIR = r"C:\Users\USER\Desktop\Projects_git\Crop yield dataset\artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # Drop Unnamed: 0 if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Rename columns as in the notebook
    df = df.rename(columns={
        'hg/ha_yield': 'Yield_hg_per_ha',
        'average_rain_fall_mm_per_year': 'Rainfall_mm',
        'pesticides_tonnes': 'Pesticides_tonnes',
        'avg_temp': 'Temperature_C'
    })
    return df

def train_and_save():
    df = load_data(DATA_PATH)
    
    # Features and Target
    X = df.drop('Yield_hg_per_ha', axis=1)
    y = df['Yield_hg_per_ha']
    
    # Preprocessing
    # Categorical columns: Area, Item
    # Numerical columns: Rainfall_mm, Pesticides_tonnes, Temperature_C, Year
    # Note: Notebook scaled everything including target. 
    # For a production pipeline, it's often better to scale features and target separately or just features.
    # I will use a Pipeline for features. For target, I'll scale it manually to keep it simple for inference.
    
    categorical_features = ['Area', 'Item']
    numerical_features = ['Year', 'Rainfall_mm', 'Pesticides_tonnes', 'Temperature_C']
    
    # Create Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    # Scale Target (optional, but following notebook's lead roughly)
    # Notebook scaled target. Let's do it too to match performance.
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)
    
    # Model (Random Forest was best)
    print("Training Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Create Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest MSE: {mse}")
    print(f"Random Forest R2: {r2}")
    
    # Feature Importance
    # Need to get feature names from OneHotEncoder
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    feature_names = numerical_features + list(ohe.get_feature_names_out(categorical_features))
    
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    # Create DataFrame for plotting
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi_df)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'feature_importance.png'))
    print("Feature importance plot saved.")
    
    # Save Model and Target Scaler
    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, 'model_pipeline.pkl'))
    joblib.dump(y_scaler, os.path.join(ARTIFACTS_DIR, 'y_scaler.pkl'))
    print("Model and scaler saved.")

if __name__ == "__main__":
    train_and_save()
