import streamlit as st
st.set_page_config(page_title="Wertgarantie Chatbot", layout="wide")

import os
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1. 文档向量初始化
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
# 2. 页面样式
# ---------------------------
st.image("https://raw.githubusercontent.com/你的用户名/你的仓库名/main/wertgarantie_logo.png", width=160)
st.markdown("""
<div style='text-align: center; margin-top: -30px;'>
    <h1>🤖 Willkommen</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .stTextInput > div > div > input {
        border-radius: 10px;
        font-size: 18px;
        padding: 10px;
    }
    .stMarkdown {
        font-size: 17px;
        line-height: 1.6;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 3. 初始化 OpenAI & 聊天记录
# ---------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "你是一个专业客服助手，可以结合公司提供的文档来回答客户的问题。"}
    ]

for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).markdown(msg["content"])

# ---------------------------
# 4. 用户输入 & 回答生成
# ---------------------------
user_input = st.chat_input("Bitte geben Sie Ihre Frage ein")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        context_chunks = get_relevant_chunks(user_input)
        context = "\n\n".join(context_chunks)
        prompt = f"""
Nutze die folgenden Informationen, um die Frage möglichst genau zu beantworten.

Context:
{context}

Frage:
{user_input}
"""
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=st.session_state.messages
        )
        reply = response.choices[0].message.content
        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        st.error(f"Fehler beim Antworten: {e}")
