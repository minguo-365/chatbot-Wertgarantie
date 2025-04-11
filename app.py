import streamlit as st
st.set_page_config(page_title="Wertgarantie Chatbot", layout="wide")

import os
import faiss
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

# 设置 OpenRouter API Key 和地址（确保你在 secrets.toml 中配置）
openai.api_key = st.secrets["OPENROUTER_API_KEY"]
openai.api_base = "https://openrouter.ai/api/v1"

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
# 2. 页面样式 & 聊天逻辑
# ---------------------------
st.title("📍 Wertgarantie 客服机器人")

st.markdown("在下方输入您的问题，我将为您提供关于保险条款、维修流程等内容的智能回复。")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("✉️ 请输入您的问题：")

if user_input:
    context = get_relevant_chunks(user_input)
    context_text = "\n".join(context)

    messages = [
        {"role": "system", "content": "你是一个专业的保险客服机器人，请根据用户问题结合上下文准确作答。"},
        {"role": "user", "content": f"以下是相关参考信息：\n{context_text}\n\n用户问题：{user_input}"}
    ]

    response = openai.ChatCompletion.create(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1:free",
        messages=messages
    )

    answer = response.choices[0].message["content"]
    st.session_state.chat_history.append((user_input, answer))
    st.markdown(f"**🤖 答复：** {answer}")

if st.session_state.chat_history:
    with st.expander("📜 查看历史对话"):
        for q, a in st.session_state.chat_history:
            st.markdown(f"- **你：** {q}\n- **AI：** {a}")
