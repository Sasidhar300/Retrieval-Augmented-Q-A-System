import streamlit as st
import requests

# Flask API URL
api_url = "http://127.0.0.1:5000/ask"

st.title("RAG-based Q&A System")
st.write("Ask questions based on the provided document corpus.")

# User input
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question:
        # Send question to backend API
        response = requests.post(api_url, json={"question": question})
        if response.status_code == 200:
            answer = response.json().get("answer", "No answer found.")
            st.write(f"**Answer:** {answer}")
        else:
            st.error("Error fetching the answer. Try again later.")
    else:
        st.warning("Please enter a question before submitting.")
