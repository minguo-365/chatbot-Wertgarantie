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
    return [(chunks[i], i) for i in I[0]]

# ---------------------------
# 2. Benutzeroberfläche & Chat-Logik
# ---------------------------
st.title("🤖 Willkommen")

# Chatverlauf löschen Button
if st.button("🧹 Verlauf löschen"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Initialisierung des Chatverlaufs
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chatverlauf anzeigen
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

# Benutzereingabe-Feld
user_input = st.chat_input("Ihre Frage eingeben:")

# Verarbeitung der Benutzereingabe
if user_input:
    st.chat_message("user").write(user_input)

    # Sonderfall: Begrüßung erkennen und sofort antworten
    if user_input.lower().strip() in ["hallo", "hi", "guten tag", "hey"]:
        welcome_reply = (
            "Hallo und willkommen bei uns! Wie kann ich für Sie helfen? Haben Sie Fragen zum Tarif, zum Angebot oder zur Anmeldung? Ich möchte gern helfen."
        )
        st.session_state.chat_history.append((user_input, welcome_reply))
        st.chat_message("assistant").write(welcome_reply)
    else:
        context = get_relevant_chunks(user_input)
        context_text = "\n".join([c[0] for c in context])
        context_source = "\n".join([f"Quelle {i+1}: Abschnitt #{cid}" for i, (_, cid) in enumerate(context)])

        # Multiround context from previous exchanges
        conversation_history = []
        for prev_user, prev_bot in st.session_state.chat_history[-6:]:  # letzte 6 Runden merken
            conversation_history.append({"role": "user", "content": prev_user})
            conversation_history.append({"role": "assistant", "content": prev_bot})

        messages = [
            {
                "role": "system",
                "content": (
                    "Sie sind ein professioneller Kundenservice-Chatbot für eine deutsche Versicherung. "
                    "Bitte antworten Sie ausschließlich auf Deutsch, in korrektem, höflichem Ton (durchgehend Siezen). "
                    "Achten Sie besonders auf Rechtschreibung, Grammatik und technische Fachbegriffe. "
                    "Ihre Antworten sollen klar, vertrauenswürdig und hilfreich sein, ohne unnötige Floskeln."
                )
            }
        ] + conversation_history + [
            {
                "role": "user",
                "content": f"Relevante Informationen:\n{context_text}\n\nFrage: {user_input}"
            }
        ]

        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=messages
        )

        answer = response.choices[0].message.content
        final_answer = answer + f"\n\n📎 Quellen:\n{context_source}"

        st.session_state.chat_history.append((user_input, final_answer))
        st.chat_message("assistant").write(final_answer)
