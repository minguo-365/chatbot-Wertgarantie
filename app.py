import streamlit as st
st.set_page_config(page_title="🤖 Willkommen", layout="wide")

import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# OpenRouter API-Schlüssel und Basis-URL setzen (in secrets.toml definiert)
client = OpenAI(api_key=st.secrets["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

# ---------------------------
# 1. Initialisierung des Vektor-Speichers für Dokumente
# ---------------------------
@st.cache_resource
def init_vector_store():
    with open("wertgarantie.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, chunks, index, embeddings

model, chunks, index, _ = init_vector_store()

def get_relevant_chunks(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]

# ---------------------------
# 2. Benutzeroberfläche & Chat-Logik
# ---------------------------
st.title("🤖 Willkommen")

if st.button("🩹 Verlauf löschen"):
    st.session_state.chat_history = []
    st.experimental_rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

user_input = st.chat_input("Ihre Frage eingeben:")

if user_input:
    st.chat_message("user").write(user_input)

    if user_input.lower().strip() in ["hallo", "hi", "guten tag", "hey"]:
        welcome_reply = (
            "Hallo und willkommen bei uns! Wie kann ich für Sie helfen? Haben Sie Fragen zum Tarif, zum Angebot oder zur Anmeldung? Ich möchte gern helfen."
        )
        st.session_state.chat_history.append((user_input, welcome_reply))
        st.chat_message("assistant").write(welcome_reply)
    else:
        context = get_relevant_chunks(user_input)
        context_text = "\n".join(context)

        conversation_history = []
        for prev_user, prev_bot in st.session_state.chat_history[-6:]:
            conversation_history.append({"role": "user", "content": prev_user})
            conversation_history.append({"role": "assistant", "content": prev_bot})

        messages = [
            {
                "role": "system",
                "content": (
                    "Sie sind ein professioneller, hilfsbereiter Kundenservice-Chatbot für eine deutsche Versicherung. "
                    "Antworten Sie stets auf Deutsch, im höflichen, professionellen Siezen-Stil. "
                    "Geben Sie bei technischen Problemen zunächst eine freundliche, hilfreiche Einschätzung oder Empfehlung. "
                    "Erwähnen Sie nur danach optional ein passendes Versicherungsangebot von WERTGARANTIE, sofern es sinnvoll ist. "
                    "Vermeiden Sie Umgangssprache oder Verwirrung. Ihre Antwort soll menschlich, kompetent und vertrauensvoll sein."
                )
            }
        ] + conversation_history + [
            {
                "role": "user",
                "content": f"Frage: {user_input}\n\nHilfreiche Informationen aus dem Dokument:\n{context_text}"
            }
        ]

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=messages
        )

        answer = response.choices[0].message.content
        st.session_state.chat_history.append((user_input, answer))
        st.chat_message("assistant").write(answer)
