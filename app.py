import streamlit as st
import joblib
from openai import OpenAI

# --- 1. CONFIGURATION ---
# OpenRouter API Key
OPENROUTER_API_KEY = "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" 
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# Load your local ML models
# Ensure these files exist in your repository
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading local ML models: {e}")

# --- 2. PREDICTION LOGIC ---
def predict_news(text):
    # Step A: Local Machine Learning Prediction
    vec = vectorizer.transform([text])
    ml_prediction = model.predict(vec)[0]
    ml_is_fake = ml_prediction == 1
    
    # Step B: OpenRouter AI Analysis (Using Gemma via OpenRouter)
    with st.spinner('🤖 Analyzing with OpenRouter Gemma...'):
        try:
            completion = client.chat.completions.create(
                model="google/gemma-3n-e2b-it:free", # Requested model
                messages=[
                    {"role": "system", "content": "You are a fact-checking assistant. Analyze the text for fake news."},
                    {"role": "user", "content": f"Analyze this headline: '{text}'. Is it TRUE or FAKE? Provide a brief reasoning."}
                ]
            )
            analysis = completion.choices[0].message.content
        except Exception as e:
            analysis = f"AI Analysis failed: {str(e)}"
    
    # Display Analysis
    st.subheader("🤖 AI Analysis Report")
    st.write(analysis)
    
    # Final Decision Logic
    is_ai_flagged = "fake" in analysis.lower() or "false" in analysis.lower()
    
    if is_ai_flagged or ml_is_fake:
        return "🟥 FAKE NEWS"
    
    return "🟩 REAL NEWS"

# --- 3. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Smart Fake News Detector", page_icon="📰")

st.title("📰 Smart Fake News Detector (OpenRouter)")
st.markdown("Powered by Local Machine Learning + OpenRouter Gemma AI Analysis.")



user_input = st.text_area("✍️ Enter news headline to analyze:", height=150)

if st.button("Analyze News"):
    if user_input:
        prediction = predict_news(user_input)
        st.header(prediction)
    else:
        st.warning("Please enter some text first.")

st.sidebar.title("📊 System Info")
st.sidebar.info("Using OpenRouter (Gemma Model) for AI analysis.")
st.markdown("---")
st.markdown("👨‍💻 Developed by Janani | Powered by OpenRouter 🚀")
