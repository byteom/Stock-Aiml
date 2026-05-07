import streamlit as st
import streamlit.components.v1 as components

def render():
    st.title("📚 API Documentation")
    st.write("Below is the interactive Swagger UI for the FastAPI backend. You can use this to execute REST endpoints directly from the browser.")
    st.info("💡 Make sure the backend server is running on port 8000: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`")
    
    st.divider()
    
    try:
        # Embed the FastAPI swagger UI using an iframe
        components.iframe("http://localhost:8000/docs", height=800, scrolling=True)
    except Exception as e:
        st.error(f"Could not load API docs. Ensure the backend is running. Error: {e}")