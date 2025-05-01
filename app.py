import streamlit as st
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# OpenRouter API-Key aus secrets laden
client = OpenAI(api_key=st.secrets["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

# ---------------------------
# 1. Initialisierung des Vektor-Speichers für Dokumente
# ---------------------------
@st.cache_resource
def init_vector_store():
    with open("wertgarantie.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, chunks, index, embeddings

embedder, chunks, index, _ = init_vector_store()

def get_relevant_chunks(query, k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [(chunks[i], i) for i in I[0]]

# ---------------------------
# 2. GLM-Modell für Tarifberechnung trainieren
# ---------------------------
@st.cache_data
def train_glm_model():
    df = pd.DataFrame({
        'Alter': [25, 45, 30, 60, 35, 22, 50],
        'Geraetewert': [800, 500, 1200, 400, 1000, 950, 350],
        'Marke': ['Apple', 'Samsung', 'Apple', 'Andere', 'Apple', 'Samsung', 'Andere'],
        'Schadenhistorie': [0, 1, 0, 1, 0, 1, 0],
        'Schadenhoehe': [0, 150, 0, 300, 0, 100, 0]
    })
    df = pd.get_dummies(df, columns=['Marke'], drop_first=True)
    formula = 'Schadenhoehe ~ Alter + Geraetewert + Schadenhistorie + Marke_Apple + Marke_Samsung'
    tweedie = sm.families.Tweedie(var_power=1.5, link=sm.families.links.log())
    glm_model = smf.glm(formula=formula, data=df, family=tweedie).fit()
    return glm_model

glm_model = train_glm_model()

# ---------------------------
# 3. Benutzeroberfläche & Chat-Logik
# ---------------------------
st.title("🧑‍💻 Wertgarantie Chatbot")

# Verlauf löschen
if st.button("🗑️ Verlauf löschen"):
    st.session_state.chat_history = []
    st.rerun()

# Session-Initialisierung
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chatverlauf anzeigen
for user_msg, bot_msg in st.session_state.chat_history:
    st.chat_message("user").write(user_msg)
    st.chat_message("assistant").write(bot_msg)

# Eingabefeld
user_input = st.chat_input("Stellen Sie Ihre Frage oder geben Sie 'Handyversicherung' ein...")

if user_input:
    st.chat_message("user").write(user_input)

    if user_input.strip().lower() == "handyversicherung":
        st.subheader("📋 Bitte beantworten Sie folgende Fragen:")

        if "frage_schritt" not in st.session_state:
            st.session_state.frage_schritt = 1

        # Schritt 1: Alter
        if st.session_state.frage_schritt == 1:
            alter = st.number_input("1️⃣ Wie alt sind Sie?", min_value=16, max_value=100, key="alter_input")
            if st.button("Weiter ➔", key="weiter1"):
                st.session_state.alter = alter
                st.session_state.frage_schritt = 2
                st.rerun()

        # Schritt 2: Gerätewert
        elif st.session_state.frage_schritt == 2:
            geraetewert = st.number_input("2️⃣ Wie viel kostet Ihr Handy? (€)", min_value=50, max_value=2000, key="wert_input")
            if st.button("Weiter ➔", key="weiter2"):
                st.session_state.geraetewert = geraetewert
                st.session_state.frage_schritt = 3
                st.rerun()

        # Schritt 3: Marke
        elif st.session_state.frage_schritt == 3:
            marke = st.selectbox("3️⃣ Welche Marke ist Ihr Handy?", ['Apple', 'Samsung', 'Andere'], key="marke_input")
            if st.button("Weiter ➔", key="weiter3"):
                st.session_state.marke = marke
                st.session_state.frage_schritt = 4
                st.rerun()

        # Schritt 4: Schadenhistorie
        elif st.session_state.frage_schritt == 4:
            schadenhistorie = st.radio("4️⃣ Gab es im letzten Jahr einen Schaden?", ['Nein', 'Ja'], key="schaden_input")
            if st.button("📊 Tarif berechnen", key="weiter4"):
                st.session_state.schadenhistorie = schadenhistorie
                st.session_state.frage_schritt = 5
                st.rerun()

        # Schritt 5: Ergebnis anzeigen
        elif st.session_state.frage_schritt == 5:
            daten = pd.DataFrame([{
                'Alter': st.session_state.alter,
                'Geraetewert': st.session_state.geraetewert,
                'Schadenhistorie': 1 if st.session_state.schadenhistorie == 'Ja' else 0,
                'Marke_Apple': 1 if st.session_state.marke == 'Apple' else 0,
                'Marke_Samsung': 1 if st.session_state.marke == 'Samsung' else 0
            }])

            erwartete_schadenhoehe = glm_model.predict(daten)[0]
            tarif_monatlich = (erwartete_schadenhoehe * 1.3) / 12

            st.success(f"✅ Ihre geschätzte monatliche Prämie beträgt: **{tarif_monatlich:.2f} €**")

            if st.button("🔄 Neue Berechnung starten"):
                del st.session_state.frage_schritt
                st.rerun()

    else:
        # Begrüßung oder Chatantwort
        if user_input.lower().strip() in ["hallo", "hi", "guten tag", "hey"]:
            welcome_reply = (
                "Hallo und herzlich willkommen bei Wertgarantie! Wie kann ich Ihnen helfen? "
                "Sie können z.\u200bB. 'Handyversicherung' eingeben oder eine Frage zu unseren Leistungen stellen."
            )
            st.chat_message("assistant").write(welcome_reply)
            st.session_state.chat_history.append((user_input, welcome_reply))
        else:
            context = get_relevant_chunks(user_input)
            context_text = "\n".join([c[0] for c in context])

            conversation_history = []
            for prev_user, prev_bot in st.session_state.chat_history[-6:]:
                conversation_history.append({"role": "user", "content": prev_user})
                conversation_history.append({"role": "assistant", "content": prev_bot})

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Du bist ein kompetenter deutscher Kundenservice-Chatbot für ein Versicherungsunternehmen. "
                        "Antworten bitte stets auf Deutsch, höflich und verständlich. Halte dich an technische und rechtliche Fakten, "
                        "aber sprich den Nutzer ruhig menschlich und freundlich an."
                    )
                }
            ] + conversation_history + [
                {"role": "user", "content": f"Relevante Inhalte:\n{context_text}\n\nFrage: {user_input}"}
            ]

            response = client.chat.completions.create(
                model="mistralai/mistral-7b-instruct:free",
                messages=messages
            )
            answer = response.choices[0].message.content
            st.chat_message("assistant").write(answer)
            st.session_state.chat_history.append((user_input, answer))
