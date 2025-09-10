import streamlit as st
import joblib
import requests
from urllib.parse import urlparse
import time
from bs4 import BeautifulSoup # Import BeautifulSoup for parsing HTML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re # Import re for cleaning text
import nltk
# from nltk.corpus import stopwords # Keep this import for stopwords
# from nltk.tokenize import word_tokenize # Keep this import for tokenization

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError: # Referencing through nltk.downloader
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError: # Referencing through nltk.downloader
    nltk.download('stopwords')

from nltk.corpus import stopwords # Import stopwords here
from nltk.tokenize import word_tokenize # Import word_tokenize here


# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Trusted sources
trusted_sources = [
    # English & Global
    "bbc.com", "reuters.com", "apnews.com", "theguardian.com", "cnn.com", "nytimes.com",
    "theatlantic.com", "economist.com", "aljazeera.com","ndtv.com",

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
    "timesofindia.indiatimes.com","thehindu.com", "indianexpress.com",

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
    "kansascity.com",

    # Taiwan
    "udn.com",

    # South Korea
    "koreajoongangdaily.joins.com"
]

# Google Search API settings
# Replace with your actual API key and Search Engine ID
API_KEY = 'AIzaSyA4T2I7q1DetLy9zhbM68KRakDsQOnoo7w' # Replace with your actual API key
SEARCH_ENGINE_ID = '37002a679147b437b' # Replace with your actual Search Engine ID

# TF-IDF Vectorizer for content comparison (using the same settings as the news model)
content_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


def google_search(query, num=5):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {'key': API_KEY, 'cx': SEARCH_ENGINE_ID, 'q': query, 'num': num}
    try:
        time.sleep(1) # Add a small delay
        res = requests.get(url, params=params)
        res.raise_for_status()  # Raise an exception for bad status codes
        results = []
        for item in res.json().get('items', []):
             results.append({
                'title': item['title'],
                'snippet': item['snippet'], # Include snippet for verification
                'url': item['link']
            })
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"Error during Google Search: {e}")
        return []


def is_trusted_url(url):
    domain = urlparse(url).netloc.lower().replace("www.", "")
    return any(trusted in domain for trusted in trusted_sources)

def clean_text(text):
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def fetch_url_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'} # Add User-Agent
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from common article tags
        paragraphs = soup.find_all(['p', 'article', 'div'], {'class': re.compile(r'content|article|body|text', re.IGNORECASE)})
        text_content = ' '.join([p.get_text() for p in paragraphs])
        if not text_content: # If no specific tags found, get all text
             text_content = soup.get_text()
        return clean_text(text_content)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None

def verify_with_sources(text, search_results):
    trusted_results = [r for r in search_results if is_trusted_url(r['url'])]
    if not trusted_results:
        return False, []

    input_cleaned = clean_text(text)
    if not input_cleaned:
        return False, []

    verified_sources = []
    corpus = [input_cleaned] # Start corpus with the input text

    # Collect cleaned content from trusted sources
    source_contents = []
    for r in trusted_results:
        content = fetch_url_content(r['url'])
        if content and len(content.split()) > 20: # Only consider sources with substantial content
            source_contents.append((r, content))
            corpus.append(content)

    if not source_contents:
        return False, []

    # Fit TF-IDF vectorizer on the combined corpus
    # We need to fit the vectorizer here because the vocabulary depends on the fetched content
    try:
        content_vectorizer.fit(corpus)
        input_vector = content_vectorizer.transform([input_cleaned])
    except ValueError as e:
         print(f"Error during TF-IDF fitting: {e}")
         return False, [] # Handle cases where corpus is empty or has issues


    # Calculate similarity
    similarity_threshold = 0.1 # Adjust this threshold based on experimentation

    for r, content in source_contents:
        source_vector = content_vectorizer.transform([content])
        similarity = cosine_similarity(input_vector, source_vector)[0][0]

        if similarity > similarity_threshold:
            verified_sources.append(r)

    return bool(verified_sources), verified_sources


def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    ml_prediction_is_fake = pred[0] == 1

    search_results = google_search(text)
    is_verified, verified_sources = verify_with_sources(text, search_results)

    st.subheader("🔎 Verification from Trusted Sources:")
    if verified_sources:
        st.write("Found supporting evidence from trusted sources:")
        for r in verified_sources:
            st.write(f"✔️ [{r['title']}]({r['url']})")
        # If trusted sources are found and ML model predicts fake,
        # consider it potentially real, but still show ML prediction.
        # If trusted sources are found and ML model predicts real,
        # confirm it as real.
        final_prediction = "🟩 REAL NEWS" if not ml_prediction_is_fake else "🟥 FAKE NEWS"
    else:
        st.write("⚠️ No strong supporting evidence found from trusted sources.")
        final_prediction = "🟥 FAKE NEWS" if ml_prediction_is_fake else "🟩 REAL NEWS"

    st.subheader("🤖 ML Model Prediction:")
    st.write("Based on the trained model:", "🟥 FAKE NEWS" if ml_prediction_is_fake else "🟩 REAL NEWS")


    return final_prediction


# Streamlit app
st.set_page_config(layout="wide", page_title="Fake News Detector", page_icon="📰") # Set favicon here

st.markdown(
    """
    <style>
    body {
         background-image: url('https://i.ytimg.com/vi/OI7b8uI2x-s/maxresdefault.jpg');
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
    user_input = st.text_area("Paste your news content or headline here...", height=200)
    if st.button("Analyze News"):
        if user_input:
            with st.spinner("Analyzing news..."):
                 prediction = predict_news(user_input)
            st.markdown(f"## {prediction}")
        else:
            st.warning("Please enter some text to analyze.")
