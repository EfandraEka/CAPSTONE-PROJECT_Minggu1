import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("ObesityDataSet.csv")

# Target
y = df["NObeyesdad"]
X = df.drop("NObeyesdad", axis=1)

# Fitur numerik dan kategorikal
numerical_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
categorical_cols = [col for col in X.columns if col not in numerical_cols]

# Preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OrdinalEncoder(), categorical_cols)
])

# Final pipeline: preprocessing + model
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model_pipeline.fit(X_train, y_train)

# Save full pipeline
joblib.dump(model_pipeline, "best_rf_model_clean.pkl")
print("Model pipeline saved to best_rf_model_clean.pkl")
