import streamlit as st
st.set_page_config(page_title="Wertgarantie Chatbot", layout="wide")

import os
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

# OpenRouter API-Schlüssel und Basis-URL setzen (in secrets.toml definiert)
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

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
st.title("📍 Wertgarantie Kundenservice-Chatbot")

st.markdown("Geben Sie unten Ihre Frage ein. Ich helfe Ihnen gerne bei Fragen zu Versicherungsbedingungen, Reparaturprozessen usw.")

# Initialisierung des Chatverlaufs
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Benutzereingabe-Feld
user_input = st.text_input("✉️ Ihre Frage eingeben:")

# Verarbeitung der Benutzereingabe
if user_input:
    context = get_relevant_chunks(user_input)
    context_text = "\n".join(context)

    messages = [
        {"role": "system", "content": "Du bist ein professioneller Kundenservice-Chatbot für eine Versicherung. Bitte beantworte Fragen der Nutzer basierend auf dem bereitgestellten Kontext sachlich und hilfreich."},
        {"role": "user", "content": f"Relevante Informationen:\n{context_text}\n\nFrage: {user_input}"}
    ]

    # Anfrage an das Sprachmodell senden
    response = openai.ChatCompletion.create(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1:free",
        messages=messages
    )

    # Antwort anzeigen und im Verlauf speichern
    answer = response.choices[0].message["content"]
    st.session_state.chat_history.append((user_input, answer))
    st.markdown(f"**🤖 Antwort:** {answer}")

# Anzeige des bisherigen Chatverlaufs
if st.session_state.chat_history:
    with st.expander("📜 Chatverlauf anzeigen"):
        for q, a in st.session_state.chat_history:
            st.markdown(f"- **Sie:** {q}\n- **Bot:** {a}")
