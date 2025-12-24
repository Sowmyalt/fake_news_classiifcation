import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("📰 Fake News Detection App")
st.subheader("Enter a news article to classify:")

user_input = st.text_area("Paste news content here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]

        if prediction == 1:
            st.success("✔ This news is REAL")
        else:
            st.error("❌ This news is FAKE")
