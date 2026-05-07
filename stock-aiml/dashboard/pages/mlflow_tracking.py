import streamlit as st
import streamlit.components.v1 as components

def render():
    st.title("🧪 MLflow Experiment Tracking")
    st.write("Monitor your TGNN and Reinforcement Learning models, view training metrics, hyperparameters, and generated artifacts.")
    st.info("💡 Make sure your MLflow tracking server is running on port 5000: `mlflow ui --port 5000`")
    
    st.divider()
    
    # Try embedding the MLflow UI
    try:
        components.iframe("http://localhost:5000", height=800, scrolling=True)
    except Exception as e:
        st.error(f"Could not load MLflow UI. Ensure MLflow is running and accessible. Error: {e}")