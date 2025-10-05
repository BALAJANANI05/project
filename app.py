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
    "india.gov.in","pmindia.gov.in","pib.gov.in","presidentofindia.gov.in",

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
    "kansascity.com","globaltimes.cn","chinadaily.com.cn","cgtn.com","scmp.com/topics/xi-jinping",

    # Taiwan
    "udn.com",

    # South Korea
    "koreajoongangdaily.joins.com"
]

# Google Search API settings
# Replace with your actual API key and Search Engine ID
API_KEY = 'AIzaSyA4T2I7q1DetLy9zhbM68KRakDsQOnoo7w' # Replace with your actual API key
SEARCH_ENGINE_ID = '37002a679147b437b' # Replace with your actual Search Engine ID


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

def verify_with_sources(text, search_results):
    trusted_results = [r for r in search_results if is_trusted_url(r['url'])]
    if not trusted_results:
        return False, []

    text_lower = text.lower()
    # A more robust check: calculate a score based on keyword overlap and presence in trusted sources
    keywords = text_lower.split()
    score = 0
    matching_sources = []

    for r in trusted_results:
        snippet_lower = r['snippet'].lower()
        title_lower = r['title'].lower()
        # Increase score for each keyword found in snippet or title
        keyword_matches = sum(word in snippet_lower or word in title_lower for word in keywords)
        score += keyword_matches
        if keyword_matches > 0: # Consider a source as "matching" if at least one keyword is found
             matching_sources.append(r)

    # Determine if verification is successful based on the score and number of matching sources
    # This threshold might need tuning based on your data and desired strictness
    verification_threshold = len(keywords) * 0.8 # Example: require 80% of keywords to be found
    is_verified = score >= verification_threshold and len(matching_sources) > 0

    return is_verified, matching_sources


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

def get_latest_news_from_source(url):
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.content, 'html.parser')

        # This is a very basic example. You would need to inspect the HTML structure
        # of each trusted news site to find the appropriate tags and classes
        # for headlines and links. This will likely be different for each site.
        headlines = soup.find_all(['h1', 'h2', 'h3', 'a'], limit=10) # Find potential headlines/links

        news_list = []
        for headline in headlines:
            text = headline.get_text(strip=True)
            link = headline.get('href')
            if text and link and urlparse(link).netloc == urlparse(url).netloc: # Only include links from the same domain
                news_list.append({'title': text, 'url': link})
        return news_list

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch news from {url}: {e}")
        return []
    except Exception as e:
        st.warning(f"Error parsing news from {url}: {e}")
        return []


def get_latest_news_from_trusted_sources(num_sources=3, num_headlines_per_source=3):
    latest_news = {}
    sources_to_check = trusted_sources[:num_sources] # Check a limited number of sources for demonstration

    for source_url in sources_to_check:
        news_from_source = get_latest_news_from_source(f"https://{source_url}")
        if news_from_source:
            latest_news[source_url] = news_from_source[:num_headlines_per_source] # Get a limited number of headlines

    return latest_news


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
            prediction = predict_news(user_input)
            st.markdown(f"## {prediction}")
        else:
            st.warning("Please enter some text to analyze.")
