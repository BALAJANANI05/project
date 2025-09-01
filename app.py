import streamlit as st
import joblib
import requests
from urllib.parse import urlparse

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


def google_search(query, num=5):
    url = 'https://www.googleapis.com/customsearch/v1'
    params = {'key': API_KEY, 'cx': SEARCH_ENGINE_ID, 'q': query, 'num': num}
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()  # Raise an exception for bad status codes
        results = []
        for item in res.json().get('items', []):
            results.append({
                'title': item['title'],
                'url': item['link']
            })
        return results
    except requests.exceptions.RequestException as e:
        st.error(f"Error during Google Search: {e}")
        return []


def is_trusted_url(url):
    domain = urlparse(url).netloc.lower().replace("www.", "")
    return any(trusted in domain for trusted in trusted_sources)


def predict_news(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    ml_prediction_is_fake = pred[0] == 1

    results = google_search(text)
    trusted = [r for r in results if is_trusted_url(r['url'])]

    st.subheader("🔎 Verified Sources:")
    if trusted:
        st.write("Found supporting evidence from trusted sources:")
        for r in trusted:
            st.write(f"✔️ [{r['title']}]({r['url']})")
        # If trusted sources are found and ML model predicts fake,
        # consider it potentially real, but still show ML prediction.
        # If trusted sources are found and ML model predicts real,
        # confirm it as real.
        final_prediction = "🟩 REAL NEWS" if not ml_prediction_is_fake else "⚠️ Potentially REAL NEWS (ML predicted FAKE)"
    else:
        st.write("⚠️ No strong supporting evidence found from trusted sources.")
        # If no trusted sources, rely solely on the ML model's prediction
        final_prediction = "🟥 FAKE NEWS" if ml_prediction_is_fake else "🟩 REAL NEWS"


    return final_prediction


# Streamlit app
st.title("📰 Smart Fake News Detector (with Google Verification)")
st.markdown("Enter a news article or headline to detect if it's REAL or FAKE using machine learning and verify with trusted Google sources.")

user_input = st.text_area("Paste your news content or headline here...", height=200)

if st.button("Analyze News"):
    if user_input:
        prediction = predict_news(user_input)
        st.markdown(f"## {prediction}")
    else:
        st.warning("Please enter some text to analyze.")
