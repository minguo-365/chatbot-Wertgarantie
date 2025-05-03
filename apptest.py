import streamlit as st
import pandas as pd
import faiss
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from openai import OpenAI
from sentence_transformers import SentenceTransformer

client = OpenAI(api_key=st.secrets["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")

@st.cache_resource
def init_vector_store():
    with open("wertgarantie.txt", "r", encoding="utf-8") as f:
        text = f.read()
    chunks = [chunk.strip() for chunk in text.split("\n\n") if len(chunk.strip()) > 50]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, chunks, index

model, chunks, index = init_vector_store()


def get_relevante_abschnitte(anfrage, k=3):
    anfrage_vektor = model.encode([anfrage])
    D, I = index.search(np.array(anfrage_vektor), k)
    return [(chunks[i], i) for i in I[0]]


def frage_openrouter(nachrichten):
    try:
        response = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct",
            messages=nachrichten
        )
        return response.choices[0].message.content
    except Exception as e:
        if "code': 402" in str(e).lower() or "insuffizien" in str(e).lower():
            try:
                response = client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct:free",
                    messages=nachrichten
                )
                return response.choices[0].message.content
            except Exception as e2:
                return f"\u274c Auch das kostenlose Modell schlug fehl: {e2}"
        return f"\u274c OpenRouter Fehler: {e}"


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

st.title("🧑‍💻 Wertgarantie Chatbot")

if st.button("🗑️ Verlauf löschen"):
    st.session_state.clear()
    st.rerun()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

USER_AVATAR = "https://avatars.githubusercontent.com/u/583231?v=4"
BOT_AVATAR = "https://img.icons8.com/emoji/48/robot-emoji.png"


def chat_bubble(inhalt, align="left", bgcolor="#F1F0F0", avatar_url=None):
    if not inhalt:
        return
    align_css = "right" if align == "right" else "left"
    avatar_html = f"<img src='{avatar_url}' style='width: 30px; height: 30px; border-radius: 50%; margin-right: 10px;' />" if avatar_url else ""
    bubble_html = f"""
        <div style='text-align: {align_css}; margin: 10px 0; display: flex; flex-direction: {'row-reverse' if align=='right' else 'row'};'>
            {avatar_html}
            <div style='background-color: {bgcolor}; padding: 10px 15px; border-radius: 10px; max-width: 80%;'>
                {inhalt}
            </div>
        </div>
    """
    st.markdown(bubble_html, unsafe_allow_html=True)


for nutzer, bot in st.session_state.chat_history:
    chat_bubble(nutzer, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
    chat_bubble(bot, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

user_input = st.chat_input("Ihre Frage eingeben:")
if user_input:
    st.chat_message("user").write(user_input)
    eingabe = user_input.strip().lower()

    if eingabe in ["hallo", "hi", "guten tag", "hey"]:
        willkommen = "Hallo und willkommen bei Wertgarantie! Was kann ich für Sie tun?"
        st.session_state.chat_history.append((user_input, willkommen))
        chat_bubble(user_input, align="right", bgcolor="#DCF8C6", avatar_url=USER_AVATAR)
        chat_bubble(willkommen, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)
    else:
        verlauf = []
        for frage, antwort in st.session_state.chat_history[-6:]:
            if frage: verlauf.append({"role": "user", "content": frage})
            verlauf.append({"role": "assistant", "content": antwort})

        context = get_relevante_abschnitte(user_input)
        context_text = "\n".join([c[0] for c in context])

        nachrichten = [
            {"role": "system", "content": (
                "Du bist ein kompetenter deutscher Kundenservice-Chatbot für ein Versicherungsunternehmen. "
                "Antworten bitte stets auf Deutsch, höflich und verständlich. Halte dich an technische und rechtliche Fakten, "
                "aber sprich den Nutzer ruhig menschlich und freundlich an."
            )}
        ] + verlauf + [
            {"role": "user", "content": f"Relevante Inhalte:\n{context_text}\n\nFrage: {user_input}"}
        ]

        antwort = frage_openrouter(nachrichten)
        st.session_state.chat_history.append((user_input, antwort))
        chat_bubble(antwort, align="left", bgcolor="#F1F0F0", avatar_url=BOT_AVATAR)

# Zusatzkategorien mit Linkausgabe
st.markdown("""---
**Wählen Sie eine Kategorie:**
""")
col1, col2 = st.columns(2)
with col1:
    if st.button("📱 Smartphone-Versicherung"):
        st.markdown("[🔍 Smartphone buchen](https://www.wertgarantie.de/versicherung/smartphone#/buchung/1)")
    if st.button("💻 Notebook-Versicherung"):
        st.markdown("[🔍 Notebook buchen](https://www.wertgarantie.de/versicherung#/notebook)")
with col2:
    if st.button("📷 Kamera-Versicherung"):
        st.markdown("[🔍 Kamera buchen](https://www.wertgarantie.de/versicherung/kamera#)")
    if st.button("📺 Fernseher-Versicherung"):
        st.markdown("[🔍 Fernseher buchen](https://www.wertgarantie.de/versicherung#/fernseher)")

if st.button("🔧 Werkstätten finden"):
    st.markdown("[🔍 Werkstattsuche](https://www.wertgarantie.de/werkstattsuche)")

if st.button("🏪 Fachhändler finden"):
    st.markdown("[🔍 Händlersuche](https://www.wertgarantie.de/haendlersuche)")

if st.button("🆘 Erste Hilfe: Handy"):
    st.markdown("[📱 Selbstreparatur Handy](https://www.wertgarantie.de/ratgeber/elektronik/smartphone/selbst-reparieren)")

if st.button("🛠️ Erste Hilfe: Haushalt"):
    st.markdown("[🏡 Selbstreparatur Haushalt](https://www.wertgarantie.de/ratgeber/elektronik/haushalt-garten/selbst-reparieren)")


