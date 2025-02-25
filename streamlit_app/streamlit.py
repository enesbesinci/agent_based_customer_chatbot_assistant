import streamlit as st
import requests
import json

st.title("Agent Based Customer Support Chatbot")
api_url = "http://localhost:2121/chat"

query = st.text_input("How can I assist you today?:")

if query:
    response = requests.post(api_url, json={"query": query})

    if response.status_code == 200:
        data = response.json()
        generated_answer = data.get("answer")

        st.subheader("Response:")
        st.write(generated_answer)
    else:
        st.error("Something went wrong! Please try again")
