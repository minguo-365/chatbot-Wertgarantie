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

# Initialisiere Modell, Inhalte, Index und Embeddings
model, chunks, index, _ = init_vector_store()

# Suche relevante Textabschnitte basierend auf Benutzeranfrage
def get_relevant_chunks(query, k=3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [chunks[i] for i in I[0]]

# ---------------------------
# 2. Benutzeroberfläche & Chat-Logik
# ---------------------------
st.title("🤖 Willkommen")

# Initialisierung des Chatverlaufs
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Benutzereingabe-Feld
user_input = st.chat_input("Ihre Frage eingeben:")

# Verarbeitung der Benutzereingabe
if user_input:
    st.chat_message("user").write(user_input)

    context = get_relevant_chunks(user_input)
    context_text = "\n".join(context)

    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein freundlicher, hilfsbereiter und professioneller Kundenservice-Chatbot "
                "für eine Versicherung. Antworte empathisch, klar und mit einer positiven Haltung. "
                "Falls du etwas nicht genau weißt, gib dein Bestes, um hilfreiche Hinweise zu geben."
            )
        },
        {"role": "user", "content": f"Relevante Informationen:\n{context_text}\n\nFrage: {user_input}"}
    ]

    # Anfrage an das Sprachmodell senden (OpenAI SDK v1 Format)
    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct:free",
        messages=messages
    )

    # Antwort anzeigen und im Verlauf speichern
    answer = response.choices[0].message.content
    st.session_state.chat_history.append((user_input, answer))
    st.chat_message("assistant").write(answer)

# Optional: Verlauf wird automatisch durch st.chat_message erhalten
