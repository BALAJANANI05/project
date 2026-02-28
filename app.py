import streamlit as st
import joblib
import requests
from urllib.parse import urlparse
import time # Import time for adding delays
from bs4 import BeautifulSoup # Import BeautifulSoup for web scraping

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl") # Load the Passive Aggressive Classifier
vectorizer = joblib.load("tfidf_vectorizer.pkl") # Load the corresponding TF-IDF vectorizer

# Trusted sources
trusted_sources = [
    # English & Global
    "bbc.com", "reuters.com", "apnews.com", "theguardian.com", "cnn.com", "nytimes.com",
    "theatlantic.com", "economist.com", "aljazeera.com","ndtv.com","today.yougov.com","abcnews.go.com","edition.cnn.com","whitehouse.gov",

    # European & Multi-language broadcasters
    "dw.com",               # Deutsche Welle
    "lemonde.fr",           # Le Monde (France)
    "elpais.com",           # El País (Spain)

    # Asia-focused
    "xinhuanet.com",        # Xinhua (China)
    "nhk.or.jp",            # NHK (Japan)
    "ptinews.com",          # Press Trust of India

    # Agencies from other regions
    "afp.com",              # Agence France-Presse
    "efe.com",              # Agencia EFE
    "anadoluagency.com",    # Anadolu Agency
    "tass.com",             # TASS (Russia)
    "ipsnews.net",          # Inter Press Service

    # Tamil-language major outlets
    "dailythanthi.com", "dinamalar.com", "dinamani.com", "malaimalar.com",
    "dinakaran.com", "tamil.thehindu.com", "thinaboomi.in", "theekkathir.in",
    "viduthalai.in", "tamilmurasu.com.sg", "thuglak.com", "ibctamil.com",

    # United States
    "nytimes.com", "washingtonpost.com", "wsj.com", "usatoday.com",
    "latimes.com", "chicagotribune.com", "bostonglobe.com",

    # United Kingdom
    "theguardian.com", "dailymail.co.uk", "thetimes.co.uk", "telegraph.co.uk",
    "independent.co.uk", "ft.com", "metro.co.uk", "mirror.co.uk",
    "express.co.uk", "thesun.co.uk",

    # Canada
    "theglobeandmail.com", "nationalpost.com", "torontostar.com",

    # India
    "timesofindia.indiatimes.com","thehindu.com", "indianexpress.com","newindianexpress.com","services.india.gov.in",
    "india.gov.in","pmindia.gov.in","pib.gov.in","presidentofindia.gov.in","en.wikipedia.org/wiki/President_of_India",
    "indiatoday.in",
    # Australia
    "smh.com.au", "theage.com.au", "afr.com", "abc.net.au",

    # Africa (Kenya)
    "nation.co.ke",

    # Singapore
    "straitstimes.com",

    # Philippines
    "inquirer.net",

    # Others
    "nypost.com", "iol.co.za", "denverpost.com", "seattletimes.com",
    "baltimoresun.com", "philly.com", "sacbee.com", "post-gazette.com",
    "kansascity.com","globaltimes.cn","chinadaily.com.cn","cgtn.com","scmp.com/topics/xi-jinping","olympics.com","worldathletics.org",
     "sports.ndtv.com","globalnews.ca","japantimes.co.jp","nknews.org","cia.gov","worldpopulationreview.com","nasa.gov","worldpopulationreview.com",
    # Taiwan
    "udn.com",

    # South Korea
    "koreajoongangdaily.joins.com","koreaboo.com","allkpop.com"
]

import streamlit as st
import joblib
from google import genai
from google.genai import types

# --- 1. CONFIGURATION & MODELS ---
# Get your free key at: https://aistudio.google.com/
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" 
client = genai.Client(api_key=GEMINI_API_KEY)

# Load your local ML models
try:
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Error loading local ML models: {e}")

# --- 2. THE AI FACT-CHECKER (Replaces old Google Search) ---
def verify_with_gemini(text):
    """Uses Gemini 2.0 + Google Search to verify news in real-time."""
    try:
        # Enable the live Google Search tool
        search_tool = types.Tool(google_search=types.GoogleSearch())
        
        # Craft a prompt for fact-checking
        prompt = f"Fact check this claim: '{text}'. Is it TRUE or FAKE? Search for recent news and give a clear verdict."
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(tools=[search_tool])
        )
        
        # Extract source links from the metadata
        sources = []
        if response.candidates[0].grounding_metadata.grounding_chunks:
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                sources.append({'title': chunk.web.title, 'url': chunk.web.uri})
                
        return response.text, sources
    except Exception as e:
        return f"AI Verification failed: {str(e)}", []

# --- 3. MAIN PREDICTION LOGIC ---
def predict_news(text):
    # Step A: Local Machine Learning Prediction
    vec = vectorizer.transform([text])
    ml_is_fake = model.predict(vec)[0] == 1 
    
    # Step B: AI Fact-Check with Google Search
    with st.spinner('🔍 Verifying with Live Google Search...'):
        analysis, verified_sources = verify_with_gemini(text)
    
    # Step C: Display Analysis
    st.subheader("🤖 AI Fact-Check Report")
    st.write(analysis)
    
    if verified_sources:
        st.write("🔗 **Evidence from Trusted Sources:**")
        for src in verified_sources:
            st.write(f"✔️ [{src['title']}]({src['url']})")
    
    # Final Decision Logic
    is_ai_flagged = "fake" in analysis.lower() or "false" in analysis.lower()
    
    if is_ai_flagged or ml_is_fake:
        # If either flags it, we alert the user
        if not is_ai_flagged and ml_is_fake:
             st.warning("⚠️ ML model flags this as suspicious, though AI found no direct debunking.")
        return "🟥 FAKE NEWS"
    
    return "🟩 REAL NEWS"

# --- 4. STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Smart Fake News Detector", page_icon="📰")

st.title("📰 Smart Fake News Detector")
st.markdown("This app uses a **Hybrid System**: Local Machine Learning + Google AI Real-time Verification.")

user_input = st.text_area("✍️ Enter news headline to analyze:", height=150)

if st.button("Analyze News"):
    if user_input:
        prediction = predict_news(user_input)
        st.header(prediction)
    else:
        st.warning("Please enter some text first.")

st.sidebar.title("📊 System Info")
st.sidebar.info("Using Gemini 2.0 Flash for real-time grounding.")
st.markdown("---")
st.markdown("👨‍💻 Developed by Janani | Powered by Google AI 🚀")

def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    ml_prediction_is_fake = pred[0] == 1

    search_results = google_search(text)
    is_verified, verified_sources = verify_with_sources(text, search_results)

    # Add a warning if the ML model predicts REAL but verification fails
    if not is_verified and not ml_prediction_is_fake:
        st.warning("⚠️ The ML model predicted REAL, but verification with trusted sources failed. Please exercise caution.")


    st.subheader("🔎 Verified Sources:")
    if verified_sources:
        st.write("Found supporting evidence from trusted sources:")
        for r in verified_sources:
            st.write(f"✔️ [{r['title']}]({r['url']})")
    else:
         st.write("⚠️ No strong supporting evidence found from trusted sources.")


    if is_verified:
        # If verification from trusted sources is successful, classify as REAL
        final_prediction = "🟩 REAL NEWS"
    else:
        # If no strong verification from trusted sources and ML model predicts fake, classify as FAKE
        final_prediction = "🟥 FAKE NEWS"


    return final_prediction

# --- ADD THIS NEW CODE ---
from google import genai
from google.genai import types

# Use your new Google AI Studio Key here
GEMINI_API_KEY = "AIzaSyBL1dzP3NsbrL01vnNBFfmLa3Whp3d0GPA" 
client = genai.Client(api_key=GEMINI_API_KEY)

def verify_with_gemini(text):
    """Uses Gemini 2.0 + Google Search to verify news in real-time."""
    try:
        # Enable the live Google Search tool
        search_tool = types.Tool(google_search=types.GoogleSearch())
        
        # Craft a prompt for fact-checking
        prompt = f"Fact check this claim: '{text}'. Is it TRUE or FAKE? Search for recent news and give a clear verdict."
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(tools=[search_tool])
        )
        
        # Extract source links from the metadata
        sources = []
        if response.candidates[0].grounding_metadata.grounding_chunks:
            for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
                sources.append({'title': chunk.web.title, 'url': chunk.web.uri})
                
        return response.text, sources
    except Exception as e:
        return f"AI Verification failed: {str(e)}", []
# --- END NEW CODE ---
# Streamlit app
st.set_page_config(layout="wide", page_title="Fake News Detector", page_icon="📰") # Set favicon here

st.markdown(
    """
    <style>
    body {
         background-image: url('https://www.istockphoto.com/illustrations/color-background');
    background-size: cover; /* cover the entire screen */
    background-repeat: no-repeat;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput textarea {
        font-size: 16px;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1 {
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .stMarkdown h2 {
        color: #555;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📰 Smart Fake News Detector (with Google Verification)")
st.markdown("Enter a news article or headline to detect if it's REAL or FAKE using machine learning and verify with trusted Google sources.")

# Use a container for input and button
with st.container():
    user_input = st.text_area("✍️ Enter or Paste your news content or headline here...", height=200)
    if st.button("🔍 Analyze News"):
        if user_input:
            prediction = predict_news(user_input)
            st.markdown(f"## {prediction}")
        else:
            st.warning("Please enter some text to analyze.")

# Sidebar (Model Info)
st.sidebar.title("📊 Model Info")
st.sidebar.write("Algorithm: Passive Aggressive Classifier")
st.sidebar.write("Technique: TF-IDF + NLP")
st.sidebar.write("Accuracy: 99.40%")

        
# FOOTER
st.markdown("---")
st.markdown("👨‍💻 Developed by BalaJanani | Powered by Machine Learning 🚀")
