from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "noise_data.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["sound_level_db"] = pd.to_numeric(df["sound_level_db"], errors="coerce")
df = df.dropna(subset=["timestamp", "location", "sound_level_db"]).copy()
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["day_sin"] = np.sin(2*np.pi*df["day_of_week"]/7)
df["day_cos"] = np.cos(2*np.pi*df["day_of_week"]/7)

numeric_features = ["sound_level_db", "hour_sin", "hour_cos", "day_sin", "day_cos"]
categorical_features = ["location"]
feature_columns = numeric_features + categorical_features
preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("location", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])
model = Pipeline([
    ("preprocessor", preprocessor),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10)),
])
model.fit(df[feature_columns])
df["cluster_id"] = model.predict(df[feature_columns])
means = df.groupby("cluster_id")["sound_level_db"].mean().sort_values()
cluster_to_category = {int(cid): name for cid, name in zip(means.index, ["Quiet Zone", "Moderate Zone", "Noisy Zone"])}
df["category"] = df["cluster_id"].map(cluster_to_category)

joblib.dump({
    "model": model,
    "cluster_to_category": cluster_to_category,
    "feature_columns": feature_columns,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "cluster_means_db": {int(k): float(v) for k, v in means.items()},
}, ARTIFACTS / "noise_kmeans_model.joblib")
df.to_csv(ROOT / "data" / "noise_data_with_categories.csv", index=False)
print("Training complete.")
print(df.groupby("category")["sound_level_db"].agg(["count", "mean", "min", "max"]))
print("Saved artifacts/noise_kmeans_model.joblib")
