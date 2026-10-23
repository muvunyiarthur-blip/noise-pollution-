from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "noise_data.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "noise_kmeans_model.joblib"

st.set_page_config(page_title="Noise Pollution Monitor", page_icon="🔊", layout="wide")
st.title("🔊 Noise Pollution Monitor")
st.caption("K-Means clustering dashboard • Quiet, Moderate, and Noisy zones")
st.warning("Starter dataset contains synthetic demonstration measurements. Replace it with real, calibrated sound-level data before using the results for decisions.")

@st.cache_resource
def load_artifact():
    if not ARTIFACT_PATH.exists():
        return None
    return joblib.load(ARTIFACT_PATH)

def time_features(ts):
    return {
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "day_sin": np.sin(2 * np.pi * ts.dayofweek / 7),
        "day_cos": np.cos(2 * np.pi * ts.dayofweek / 7),
    }

artifact = load_artifact()
if artifact is None:
    st.error("Model file not found. Run all cells in notebooks/noise_monitor_step_by_step.ipynb to create artifacts/noise_kmeans_model.joblib, then restart the app.")
    st.stop()

model = artifact["model"]
cluster_to_category = {int(k): v for k, v in artifact["cluster_to_category"].items()}
feature_columns = artifact["feature_columns"]

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data["sound_level_db"] = pd.to_numeric(data["sound_level_db"], errors="coerce")
    data = data.dropna(subset=["timestamp", "location", "sound_level_db"])
else:
    data = pd.DataFrame(columns=["timestamp", "location", "sound_level_db"])

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Predict a zone", "📁 Upload data"])

with tab1:
    if not data.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Readings", f"{len(data):,}")
        c2.metric("Locations", str(data["location"].nunique()))
        c3.metric("Average sound level", f"{data['sound_level_db'].mean():.1f} dB")
        plot_data = data.copy()
        plot_data["hour"] = plot_data["timestamp"].dt.hour
        plot_data["day_of_week"] = plot_data["timestamp"].dt.dayofweek
        plot_data["hour_sin"] = np.sin(2 * np.pi * plot_data["hour"] / 24)
        plot_data["hour_cos"] = np.cos(2 * np.pi * plot_data["hour"] / 24)
        plot_data["day_sin"] = np.sin(2 * np.pi * plot_data["day_of_week"] / 7)
        plot_data["day_cos"] = np.cos(2 * np.pi * plot_data["day_of_week"] / 7)
        plot_data["cluster_id"] = model.predict(plot_data[feature_columns])
        plot_data["category"] = plot_data["cluster_id"].map(cluster_to_category)
        left, right = st.columns(2)
        with left:
            st.subheader("Average sound level by location")
            st.bar_chart(plot_data.groupby("location")["sound_level_db"].mean().sort_values())
        with right:
            st.subheader("Zone distribution")
            st.bar_chart(plot_data["category"].value_counts())
        st.subheader("Recent readings with predicted zones")
        st.dataframe(plot_data[["timestamp", "location", "sound_level_db", "category"]].sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("Download categorized readings (CSV)", plot_data.to_csv(index=False).encode("utf-8"), "noise_readings_with_zones.csv", "text/csv")
    else:
        st.info("No data CSV found. Use the Upload data tab to analyze a CSV.")

with tab2:
    st.subheader("Predict a new sound reading")
    locations = sorted(set(data["location"].dropna().astype(str).tolist())) if not data.empty else []
    location = st.selectbox("Location", locations + ["Other / new location"])
    if location == "Other / new location":
        location = st.text_input("Enter location name", "New Location")
    sound_level = st.slider("Sound level (dB)", min_value=20.0, max_value=120.0, value=60.0, step=1.0)
    reading_date = st.date_input("Measurement date", value=pd.Timestamp.now().date())
    reading_time = st.time_input("Measurement time", value=pd.Timestamp.now().time().replace(second=0, microsecond=0))
    if st.button("Classify noise zone", type="primary"):
        ts = pd.Timestamp.combine(reading_date, reading_time)
        row = {"sound_level_db": float(sound_level), "location": str(location), **time_features(ts)}
        sample = pd.DataFrame([row])[feature_columns]
        cluster = int(model.predict(sample)[0])
        zone = cluster_to_category[cluster]
        st.subheader(f"Result: {zone}")
        st.write(f"**Location:** {location}  \n**Sound level:** {sound_level:.1f} dB  \n**Time:** {ts}")
        if zone == "Quiet Zone":
            st.success("This reading falls in the quietest cluster learned from the training data.")
        elif zone == "Moderate Zone":
            st.info("This reading falls in the middle cluster learned from the training data.")
        else:
            st.error("This reading falls in the highest-sound cluster learned from the training data.")
        st.caption("Cluster names are relative to the training dataset, not a regulatory noise compliance decision.")

with tab3:
    st.subheader("Upload your own measurements")
    uploaded = st.file_uploader("CSV must include timestamp, location, sound_level_db", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            required = {"timestamp", "location", "sound_level_db"}
            if not required.issubset(uploaded_df.columns):
                st.error("Missing columns: " + ", ".join(sorted(required - set(uploaded_df.columns))))
            else:
                uploaded_df["timestamp"] = pd.to_datetime(uploaded_df["timestamp"], errors="coerce")
                uploaded_df["sound_level_db"] = pd.to_numeric(uploaded_df["sound_level_db"], errors="coerce")
                uploaded_df = uploaded_df.dropna(subset=["timestamp", "location", "sound_level_db"]).copy()
                uploaded_df["hour_sin"] = uploaded_df["timestamp"].dt.hour.map(lambda h: np.sin(2 * np.pi * h / 24))
                uploaded_df["hour_cos"] = uploaded_df["timestamp"].dt.hour.map(lambda h: np.cos(2 * np.pi * h / 24))
                uploaded_df["day_sin"] = uploaded_df["timestamp"].dt.dayofweek.map(lambda d: np.sin(2 * np.pi * d / 7))
                uploaded_df["day_cos"] = uploaded_df["timestamp"].dt.dayofweek.map(lambda d: np.cos(2 * np.pi * d / 7))
                uploaded_df["cluster_id"] = model.predict(uploaded_df[feature_columns])
                uploaded_df["category"] = uploaded_df["cluster_id"].map(cluster_to_category)
                st.success(f"Classified {len(uploaded_df)} valid readings.")
                st.dataframe(uploaded_df[["timestamp", "location", "sound_level_db", "category"]], use_container_width=True)
                st.download_button("Download uploaded data with zones", uploaded_df.to_csv(index=False).encode("utf-8"), "uploaded_noise_zones.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")

st.divider()
st.caption("For research and demonstration. Calibrate your sound meter and validate cluster thresholds before operational use.")
