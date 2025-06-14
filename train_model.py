import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("ObesityDataSet.csv")

# Fitur kategorikal yang perlu encoding
categorical_cols = ['Gender', 'CALC', 'FAVC', 'SCC', 'SMOKE',
                    'family_history_with_overweight', 'CAEC', 'MTRANS']

# Simpan encoder
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Encode target label
target_encoder = LabelEncoder()
df["NObeyesdad"] = target_encoder.fit_transform(df["NObeyesdad"])

# Fitur & target
X = df.drop("NObeyesdad", axis=1)
y = df["NObeyesdad"]

# Latih model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Simpan semua dalam satu dict
joblib.dump({
    "model": model,
    "encoders": encoders,
    "target_encoder": target_encoder,
    "feature_names": X.columns.tolist()
}, "model.pkl")

print(" Model berhasil disimpan ke model.pkl")
