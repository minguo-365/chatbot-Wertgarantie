import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
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
# -------- Modelltraining (kann später ausgelagert werden) --------
@st.cache_data
def train_model():
    data = pd.DataFrame({
        'Alter': [25, 45, 30, 60, 35, 22, 50],
        'Geraetewert': [800, 500, 1200, 400, 1000, 950, 350],
        'Marke': ['Apple', 'Samsung', 'Apple', 'Andere', 'Apple', 'Samsung', 'Andere'],
        'Schadenhistorie': [0, 1, 0, 1, 0, 1, 0],
        'Schadenhoehe': [0, 150, 0, 300, 0, 100, 0]
    })
    data = pd.get_dummies(data, columns=['Marke'], drop_first=True)
    formula = 'Schadenhoehe ~ Alter + Geraetewert + Schadenhistorie + Marke_Apple + Marke_Samsung'
    tweedie = sm.families.Tweedie(var_power=1.5, link=sm.families.links.log())
    model = smf.glm(formula=formula, data=data, family=tweedie).fit()
    return model

model = train_model()
    

# ---------------------------
# 2. Benutzeroberfläche & Chat-Logik
# ---------------------------
st.title("🤖 Willkommen")

# Chatverlauf löschen Button
if st.button("🩹 Verlauf löschen"):
    st.session_state.chat_history = []
    st.rerun()

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

    if user_input.strip().lower() == "handyversicherung":
        st.subheader("📱 Bitte geben Sie Ihre Gerätedaten ein:")

        alter = st.number_input("Wie alt sind Sie?", min_value=16, max_value=100, value=30)
        geraetewert = st.number_input("Wie viel kostet Ihr Handy? (€)", min_value=50, max_value=2000, value=900)
        marke = st.selectbox("Welche Marke ist Ihr Handy?", ['Apple', 'Samsung', 'Andere'])
        schadenhistorie = st.radio("Gab es im letzten Jahr einen Schaden?", ['Nein', 'Ja'])
        schadenhistorie_code = 1 if schadenhistorie == 'Ja' else 0

        if st.button("📊 Tarif berechnen"):
            daten = pd.DataFrame([{
                'Alter': alter,
                'Geraetewert': geraetewert,
                'Schadenhistorie': schadenhistorie_code,
                'Marke_Apple': 1 if marke == 'Apple' else 0,
                'Marke_Samsung': 1 if marke == 'Samsung' else 0
            }])

            erwartete_schadenhoehe = model.predict(daten)[0]
            tarif_monatlich = (erwartete_schadenhoehe * 1.3) / 12
            st.success(f"✅ Ihre geschätzte monatliche Prämie beträgt: **{tarif_monatlich:.2f} €**")

    else:
        st.info("💡 Bitte geben Sie **HandyVersicherung** ein, um die Tarifberechnung zu starten.")

    
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
                    "Ihre Antworten sollen klar, vertrauenswürdig und hilfreich sein. Geben Sie, falls relevant, zuerst eine hilfreiche Einschätzung und schlagen Sie danach passend ein Versicherungsprodukt vor."
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

        st.session_state.chat_history.append((user_input, answer))
        st.chat_message("assistant").write(answer)
