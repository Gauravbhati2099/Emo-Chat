import streamlit as st
from transformers import pipeline

# Load the emotion classification model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

emotion_classifier = load_model()

# Get emotion with additional descriptive response
def get_emotion_with_response(text):
    result = emotion_classifier(text)
    emotion = result[0][0]["label"]
    score = result[0][0]["score"]
    
    # Add a customized response for each emotion
    emotion_response = {
        "anger": "You seem really upset or frustrated. There's a lot of intense feeling here.",
        "fear": "It sounds like you're expressing anxiety or fear about something.",
        "joy": "You're radiating positivity and happiness! 😊",
        "sadness": "It seems like you're going through a tough time. Stay strong.",
        "surprise": "This is unexpected! Looks like something caught you off guard.",
        "neutral": "You're in a calm state, no strong emotions detected.",
        "love": "There is a sense of affection and deep connection here."
    }
    
    emotion_description = emotion_response.get(emotion, "This text is quite complex. Emotion could be mixed.")
    
    return f"**Predicted Emotion:** {emotion} (Confidence: {score:.2f})\n{emotion_description}"

# Streamlit UI
st.title("🧠 Emotion Detection from Text")
st.markdown("This app uses a fine-tuned DistilRoBERTa model to classify the emotion in your text.")

user_input = st.text_area("💬 Enter your text here:")

if st.button("🚀 Detect Emotion"):
    if user_input:
        emotion_info = get_emotion_with_response(user_input)
        st.success(emotion_info)
    else:
        st.warning("⚠️ Please enter some text.")

st.caption("Made with 🤗 Transformers & Streamlit")
