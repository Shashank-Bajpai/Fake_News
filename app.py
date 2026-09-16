import joblib
import streamlit as st
from preprocess import clean_hindi_text

st.set_page_config(page_title="हिंदी फेक न्यूज़ डिटेक्टर", page_icon="📰")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.joblib")
    vectorizer = joblib.load("vectorizer.joblib")
    metadata = joblib.load("metadata.joblib")
    return model, vectorizer, metadata

model, vectorizer, metadata = load_artifacts()

st.title("📰 हिंदी फेक न्यूज़ डिटेक्टर")
st.caption(f"Domain-held-out test accuracy: {metadata['accuracy']:.1%} | F1: {metadata['f1']:.1%}")

text = st.text_area("यहाँ खबर पेस्ट करें (हिंदी में):", height=200)

if st.button("जांच करें", type="primary"):
    if not text.strip():
        st.warning("कृपया कुछ टेक्स्ट डालें।")
    else:
        cleaned = clean_hindi_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("🚩 यह खबर फर्जी (FAKE) हो सकती है")
        else:
            st.success("✅ यह खबर वास्तविक (REAL) लगती है")

        with st.expander("मॉडल को दिया गया साफ़ टेक्स्ट देखें"):
            st.code(cleaned)

        st.info("⚠️ यह मॉडल फैक्ट-चेक शैली बनाम सामान्य समाचार शैली की पहचान करता है — यह दावों की सच्चाई स्वयं सत्यापित नहीं करता।")