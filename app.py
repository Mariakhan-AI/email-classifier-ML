import streamlit as st
import pickle
import re
import string
from nltk.stem.porter import PorterStemmer

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()

# ----------------------------
# Built-in stopwords (no NLTK downloads required)
# ----------------------------
STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","any","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now",
    "d","ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn",
    "hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn"
}

# ----------------------------
# Text Preprocessing (no nltk tokenizers)
# ----------------------------
def simple_tokenize(text):
    # Lowercase and extract words (alphanumeric)
    text = text.lower()
    # Use regex to match word tokens (keeps alphanumeric)
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens

def transform_text(text):
    # 1) tokenize
    tokens = simple_tokenize(text)

    # 2) remove stopwords and punctuation and keep alphanumeric only
    filtered = [t for t in tokens if t not in STOPWORDS]

    # 3) stemming
    stemmed = [ps.stem(t) for t in filtered]

    return " ".join(stemmed)

# ----------------------------
# Streamlit Page Settings
# ----------------------------
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🚨",
    layout="wide"
)

# ----------------------------
# Custom CSS (Dark + Animations)
# ----------------------------
st.markdown("""
    <style>

    /* Main dark background */
    .main {
        background-color: #0d1117;
        color: white;
        animation: fadeIn 1.2s ease-in-out;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
        animation: slideRight 1s ease;
    }

    /* Input fields */
    .stTextArea textarea {
        background-color: #161b22 !important;
        color: white !important;
        border-radius: 10px;
    }

    /* Smooth fade animation */
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }

    /* Slide-in animation */
    @keyframes slideRight {
        0% {transform: translateX(-50px); opacity: 0;}
        100% {transform: translateX(0); opacity: 1;}
    }

    /* Animated result box */
    .result-box {
        padding: 20px;
        border-radius: 12px;
        font-size: 24px;
        font-weight: bold;
        animation: popIn .5s ease-out;
    }

    @keyframes popIn {
        0% {transform: scale(0.5); opacity: 0;}
        100% {transform: scale(1); opacity: 1;}
    }

    /* Minor layout tweaks */
    .stButton>button {
        background: linear-gradient(90deg,#16a34a,#059669);
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }

    </style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("📌 Navigation")
st.sidebar.markdown("### Choose an option:")
st.sidebar.write("🔹 Spam Detection Tool")  
st.sidebar.write("🔹 NLP + ML Model")  
st.sidebar.write("🔹 Built with Streamlit")  

st.sidebar.markdown("----")
st.sidebar.success("✨ ML Spam Detector is LIVE!")

# ----------------------------
# Main Title
# ----------------------------
st.markdown("<h1 style='color:white;'>💬 Spam Message Detector (ML + NLP)</h1>", unsafe_allow_html=True)
st.markdown("### 🚀 Fast, Accurate & Clean — Powered by Machine Learning")

# ----------------------------
# User Input
# ----------------------------
user_input = st.text_area(
    "📥 Enter a message",
    height=140,
    placeholder="Type something like: 'You won a lottery of $1,000,000!'"
)

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("🔍 Analyze Message"):

    if len(user_input.strip()) == 0:
        st.warning("⚠️ Please type a message first!")
    else:
        transformed = transform_text(user_input)
        vector_input = vectorizer.transform([transformed]).toarray()
        result = model.predict(vector_input)[0]

        # Animated result
        if result == 1:
            st.markdown(
                "<div class='result-box' style='background:#ff4d4d; color:white;'>🚨 SPAM DETECTED</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background:#22c55e; color:white;'>✅ NOT SPAM</div>",
                unsafe_allow_html=True
            )

# Footer
st.write("----")
st.markdown(
    "<p style='text-align:center; color: gray;'>🔥 Built with Machine Learning, NLP, and Streamlit • Styled with ❤️</p>",
    unsafe_allow_html=True,
)

