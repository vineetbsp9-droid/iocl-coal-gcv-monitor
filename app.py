import time
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from database import init_db, log_prediction, get_logs
from pdf_generator import generate_sample_pdf

# Initialize database
init_db()

@st.cache_resource
def load_ml_model():
    return joblib.load('coal_gcv_model.pkl')

model = load_ml_model()

st.set_page_config(page_title="IOCL Coal & Petcoke Quality Monitor", layout="wide")

def generate_live_sensor_reading(rng):
    """Simulate real-time readings from an online coal/petcoke analyzer."""
    moisture = round(float(rng.uniform(4.0, 16.0)), 2)
    ash = round(float(rng.uniform(10.0, 32.0)), 2)
    volatile = round(float(rng.uniform(18.0, 34.0)), 2)
    fixed_carbon = round(max(0.0, 100.0 - (moisture + ash + volatile)), 2)
    sulfur = round(float(rng.uniform(0.5, 4.5)), 2)
    return moisture, ash, volatile, fixed_carbon, sulfur

def classify_fuel(gcv, moisture, sulfur, target_gcv, petcoke_gcv, max_sulfur_limit):
    """Evaluate fuel grade and compute petcoke blending recommendations."""
    blending_note = "Fuel meets standard thermal efficiency criteria. Direct feeding to boiler/gasifier approved."
    if gcv >= target_gcv and moisture <= 10:
        grade = "Grade A (Direct Use)"
        status_type = "success"
    elif gcv >= 4200:
        grade = "Grade B (Blend Required)"
        status_type = "warning"
        if gcv < target_gcv and petcoke_gcv > gcv:
            petcoke_pct = ((target_gcv - gcv) / (petcoke_gcv - gcv)) * 100
            petcoke_pct = min(max(petcoke_pct, 0.0), 100.0)
            coal_pct = 100.0 - petcoke_pct
            est_blend_sulfur = (coal_pct / 100) * sulfur + (petcoke_pct / 100) * 5.5
            blending_note = (
                f"Recommended Blending Ratio: Mix {coal_pct:.1f}% Coal with {petcoke_pct:.1f}% Petcoke "
                f"to achieve target {target_gcv} kcal/kg. Estimated Blend Sulfur: {est_blend_sulfur:.2f}%."
            )
    else:
        grade = "Grade C (Reject / Dry)"
        status_type = "error"
        blending_note = "CRITICAL ALERT: High moisture or low calorific value detected. Route coal to dryer unit or reject lot."
    return grade, status_type, blending_note

st.title("🏭 IOCL Coal & Petcoke GCV Predictor & Blending Optimizer")
st.markdown("Real-time Fuel Quality Analysis, Database Logging & Explainable AI for Boiler & Gasification Units")

# Sidebar Configuration
st.sidebar.header("⚙️ Plant Target Settings")
target_gcv = st.sidebar.number_input("Target Boiler GCV (kcal/kg)", value=5500, step=100)
petcoke_gcv = st.sidebar.number_input("Available Petcoke GCV (kcal/kg)", value=8100, step=100)
max_sulfur_limit = st.sidebar.number_input("Max Allowable Blend Sulfur (%)", value=2.5, step=0.1)

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎛️ Single Sample & Blending",
    "📂 Batch CSV Upload",
    "🧠 AI Explainability (SHAP)",
    "📊 Plant Logs & History",
    "⚡ Real-Time Live Stream"
])

with tab1:
    st.subheader("Enter Sample Parameters:")
    col1, col2 = st.columns(2)

    with col1:
        moisture = st.slider("Moisture (%)", 1.0, 25.0, 8.0)
        ash = st.slider("Ash (%)", 2.0, 45.0, 18.0)
        sulfur = st.slider("Sulfur (%)", 0.1, 7.0, 1.2)

    with col2:
        volatile = st.slider("Volatile Matter (%)", 10.0, 40.0, 25.0)
        fixed_carbon = round(100 - (moisture + ash + volatile), 2)
        st.metric(label="Calculated Fixed Carbon (%)", value=f"{fixed_carbon}%")

    if fixed_carbon < 0:
        st.error("⚠️ Error: The sum of Moisture, Ash, and Volatile Matter must be less than 100%!")
    else:
        if st.button("Analyze GCV & Quality"):
            input_data = np.array([[moisture, ash, volatile, fixed_carbon, sulfur]])
            predicted_gcv = float(model.predict(input_data)[0])
            grade, status_type, blending_note = classify_fuel(
                predicted_gcv, moisture, sulfur, target_gcv, petcoke_gcv, max_sulfur_limit
            )

            if status_type == "success":
                st.success(f"✅ Optimal Quality ({grade}): Suitable for direct feeding into the boiler/gasifier.")
            elif status_type == "warning":
                st.warning(f"⚠️ Moderate Quality ({grade}): Blend with High-GCV Petcoke to improve thermal efficiency.")
                if predicted_gcv < target_gcv and petcoke_gcv > target_gcv:
                    petcoke_pct = ((target_gcv - predicted_gcv) / (petcoke_gcv - predicted_gcv)) * 100
                    petcoke_pct = min(max(petcoke_pct, 0.0), 100.0)
                    coal_pct = 100.0 - petcoke_pct
                    est_blend_sulfur = (coal_pct / 100) * sulfur + (petcoke_pct / 100) * 5.5
                    st.info(f"🧪 **Petcoke Blending Optimizer:** {blending_note}")
                    if est_blend_sulfur > max_sulfur_limit:
                        st.error(f"⚠️ Warning: Estimated blend sulfur ({est_blend_sulfur:.2f}%) exceeds plant SOx limit ({max_sulfur_limit:.2f}%)!")
            else:
                st.error(f"🚨 Low Quality / High Moisture ({grade}): Dry coal before use or reject batch.")

            st.divider()
            st.subheader(f"🔥 Predicted GCV: {predicted_gcv:.2f} kcal/kg")

            log_prediction(moisture, ash, volatile, fixed_carbon, sulfur, predicted_gcv, grade)
            st.session_state['last_input'] = input_data

            pdf_bytes = generate_sample_pdf(
                moisture, ash, volatile, fixed_carbon, sulfur,
                predicted_gcv, grade, blending_note
            )
            st.download_button(
                label="📄 Download Official Audit Report (PDF)",
                data=pdf_bytes,
                file_name="IOCL_Fuel_Audit_Report.pdf",
                mime="application/pdf"
            )

with tab2:
    st.subheader("Bulk Sample CSV Upload")
    uploaded_file = st.file_uploader(
        "Upload CSV File (Columns: Moisture, Ash, Volatile_Matter, Fixed_Carbon, Sulfur)",
        type=["csv"]
    )
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        req_cols = ['Moisture', 'Ash', 'Volatile_Matter', 'Fixed_Carbon', 'Sulfur']
        if all(col in batch_df.columns for col in req_cols):
            batch_df['Predicted_GCV'] = np.round(model.predict(batch_df[req_cols]), 2)
            st.dataframe(batch_df, use_container_width=True)

            csv_export = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Analyzed Batch Report (CSV)",
                data=csv_export,
                file_name="iocl_batch_gcv_report.csv",
                mime="text/csv"
            )
        else:
            st.error(f"CSV must contain the following columns: {req_cols}")

with tab3:
    st.subheader("🧠 Model Interpretability using SHAP")
    st.markdown("This chart illustrates how each feature (e.g., Ash, Moisture, Fixed Carbon) contributed to the predicted GCV for the current sample.")

    if 'last_input' in st.session_state:
        feature_names = ['Moisture', 'Ash', 'Volatile_Matter', 'Fixed_Carbon', 'Sulfur']
        input_df = pd.DataFrame(st.session_state['last_input'], columns=feature_names)

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(input_df)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.bar(shap_values[0], max_display=5, show=False)
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
    else:
        st.warning("⚠️ Please go to the 'Single Sample & Blending' tab first, set parameters, and click 'Analyze GCV & Quality' to view the SHAP explanation here.")

with tab4:
    st.subheader("📊 Plant Prediction History & Logs")
    st.markdown("Historical audit trail of all fuel quality tests and predictions stored in the plant database:")

    logs_df = get_logs()
    if not logs_df.empty:
        st.dataframe(logs_df, use_container_width=True)

        csv_logs = logs_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Database Logs (CSV)",
            data=csv_logs,
            file_name="iocl_plant_prediction_logs.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction logs available in the database yet.")

with tab5:
    st.subheader("⚡ Real-Time Online Analyzer Stream (Simulated)")
    st.markdown("Continuously polls simulated conveyor/boiler feed analyzer readings, evaluates quality grades, triggers alerts, and logs readings to SQLite.")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.5, 1, 1])
    with ctrl_col1:
        live_enabled = st.toggle("Enable Live Sensor Stream", value=False)
    with ctrl_col2:
        poll_interval = st.slider("Poll Interval (seconds)", 1, 10, 3)
    with ctrl_col3:
        num_cycles = st.slider("Stream Cycles per Run", 5, 50, 15)

    stream_placeholder = st.empty()

    if "stream_rng" not in st.session_state:
        st.session_state["stream_rng"] = np.random.default_rng(42)
    if "stream_history" not in st.session_state:
        st.session_state["stream_history"] = []
    if "stream_tick_counter" not in st.session_state:
        st.session_state["stream_tick_counter"] = 0

    if live_enabled:
        for _ in range(num_cycles):
            st.session_state["stream_tick_counter"] += 1
            tick_id = st.session_state["stream_tick_counter"]

            m, a, vm, fc, s = generate_live_sensor_reading(st.session_state["stream_rng"])
            gcv = float(model.predict(np.array([[m, a, vm, fc, s]]))[0])
            grade, status_type, blending_note = classify_fuel(
                gcv, m, s, target_gcv, petcoke_gcv, max_sulfur_limit
            )

            log_prediction(m, a, vm, fc, s, gcv, grade)
            st.session_state["last_input"] = np.array([[m, a, vm, fc, s]])

            st.session_state["stream_history"].insert(0, {
                "Tick": tick_id,
                "Moisture (%)": m,
                "Ash (%)": a,
                "Volatile Matter (%)": vm,
                "Fixed Carbon (%)": fc,
                "Sulfur (%)": s,
                "Predicted GCV (kcal/kg)": round(gcv, 2),
                "Grade": grade
            })
            st.session_state["stream_history"] = st.session_state["stream_history"][:25]

            with stream_placeholder.container():
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Live GCV", f"{gcv:.2f} kcal/kg")
                k2.metric("Moisture / Ash", f"{m:.1f}% / {a:.1f}%")
                k3.metric("Sulfur", f"{s:.2f}%")
                k4.metric("Classification", grade)

                if status_type == "error":
                    st.error(f"🚨 Live Feed Alert (Tick #{tick_id}): {blending_note}")
                elif status_type == "warning":
                    st.warning(f"⚠️ Blending Advisory (Tick #{tick_id}): {blending_note}")
                else:
                    st.success(f"✅ Normal Feed (Tick #{tick_id}): {blending_note}")

                st.dataframe(pd.DataFrame(st.session_state["stream_history"]), use_container_width=True)

            time.sleep(poll_interval)
    else:
        st.info("Toggle **Enable Live Sensor Stream** above to start streaming simulated online analyzer readings.")
        if st.session_state["stream_history"]:
            st.dataframe(pd.DataFrame(st.session_state["stream_history"]), use_container_width=True)