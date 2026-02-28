import streamlit as st
import joblib
from google import genai
from google.genai import types
import time

# --- 1. CONFIGURATION & MODELS ---
# Get your free key at: https://aistudio.google.com/
GEMINI_API_KEY = "AIzaSyD7Wfwd2_34GnbOps0G45MT0dKOy_V6KrI" 
client = genai.Client(api_key=GEMINI_API_KEY)

# Load local ML models
try:
    model = joblib.load("fake_news_model.pkl") # Passive Aggressive Classifier
    vectorizer = joblib.load("tfidf_vectorizer.pkl") # TF-IDF vectorizer
except Exception as e:
    st.error(f"Error loading local ML models: {e}")

# --- 2. AI FACT-CHECKER (Replaces old Google Search) ---
def verify_with_gemini(text):
    """Uses Gemini 2.0 + Google Search to verify news in real-time."""
    max_retries = 2
    for attempt in range(max_retries):
        try:
            search_tool = types.Tool(google_search=types.GoogleSearch())
            prompt = f"Fact check this claim: '{text}'. Is it TRUE or FAKE? Search for recent news and give a clear verdict."
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(tools=[search_tool])
            )
            
            sources = []
            if response.candidates[0].grounding_metadata.grounding_chunks:
                for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                    sources.append({'title': chunk.web.title, 'url': chunk.web.uri})
                    
            return response.text, sources
        
        except Exception as e:
            if "429" in str(e):
                st.warning(f"Quota exceeded. Waiting 20 seconds before retrying (Attempt {attempt+1}/{max_retries})...")
                time.sleep(20) # Wait and retry
            else:
                return f"AI Verification failed: {str(e)}", []
    
    return "AI Verification failed: Quota limit reached.", []

# --- 3. MAIN PREDICTION LOGIC ---
def predict_news(text):
    # Step A: Local Machine Learning Prediction
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    ml_prediction_is_fake = pred[0] == 1
    
    # Step B: Real-Time Fact-Check with Gemini
    # This bypasses the 403 error
    with st.spinner('🔍 Verifying with Live Google Search...'):
        analysis, verified_sources = verify_with_gemini(text)
    
    # Step C: Display Detailed AI Analysis
    st.subheader("🤖 AI Fact-Check Report")
    st.write(analysis)
    
    if verified_sources:
        st.write("🔗 **Evidence from Trusted Sources:**")
        for src in verified_sources:
            st.write(f"✔️ [{src['title']}]({src['url']})")
    
    # Step D: Final Decision Logic
    is_ai_flagged = "fake" in analysis.lower() or "false" in analysis.lower()
    
    if is_ai_flagged or ml_prediction_is_fake:
        # If either flags it, alert the user
        if not is_ai_flagged and ml_prediction_is_fake:
             st.warning("⚠️ ML model flags this as suspicious, though AI found no direct debunking.")
        return "🟥 FAKE NEWS"
    
    return "🟩 REAL NEWS"

# --- 4. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Smart Fake News Detector", page_icon="📰")

st.title("📰 Smart Fake News Detector (with Gemini AI)")
st.markdown("Enter a news article or headline to detect if it's REAL or FAKE using machine learning and verify with Google AI.")

# Input area
user_input = st.text_area("✍️ Enter or Paste your news content or headline here...", height=200)

if st.button("🔍 Analyze News"):
    if user_input:
        prediction = predict_news(user_input)
        st.markdown(f"## {prediction}")
    else:
        st.warning("Please enter some text to analyze.")

# Sidebar (Model Info)
st.sidebar.title("📊 System Info")
st.sidebar.write("Algorithm: Passive Aggressive Classifier")
st.sidebar.write("Technique: TF-IDF + Gemini Grounding")
st.sidebar.info("Real-time verification powered by Gemini 2.0 Flash.")
        
# FOOTER
st.markdown("---")
st.markdown("👨‍💻 Developed by Janani | Powered by Google AI 🚀")
