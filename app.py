import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit_mic_recorder import speech_to_text
import re

##### ------------------- CONFIG -------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

####------------------- CUSTOM CSS -------------------
st.markdown("""
    <style>
    body {
        font-family: "Inter", sans-serif;
        background-color: #f9fafb;
    }
    .chat-container {
        max-height: 75vh;
        overflow-y: auto;
        padding: 10px;
        margin-bottom: 90px;
    }
    .user-bubble {
        background: #DCF8C6;
        padding: 12px 16px;
        border-radius: 18px 18px 0 18px;
        margin: 8px 0;
        display: inline-block;
        max-width: 75%;
        float: right;
        clear: both;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .assistant-bubble {
        background: #ffffff;
        border: 1px solid #e5e5e5;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 0;
        margin: 8px 0;
        display: inline-block;
        max-width: 75%;
        float: left;
        clear: both;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.08);
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 300px;
        width: calc(100% - 300px);
        border-top: 1px solid #ddd;
        background: white;
        padding: 12px 20px;
        z-index: 999;
    }
    @media (max-width: 768px) {
        .stChatInput {
            left: 0 !important;
            width: 100% !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- SESSION STATE -------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi, I’m your AI assistant. Upload a PDF or ask me anything!"}
    ]
if "pdf_retriever" not in st.session_state:
    st.session_state["pdf_retriever"] = None
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatGroq(
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        streaming=True
    )

# ------------------- TOOLS -------------------
if "tools" not in st.session_state:
    arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000))
    search = DuckDuckGoSearchRun(name="Search", description="Get real-time information, news, or updates.")
    st.session_state["tools"] = [search, arxiv, wiki]

# ------------------- SIDEBAR -------------------
with st.sidebar:
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Upload your document", type=["pdf"])
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = [
            {"role": "assistant", "content": "👋 Hi, I’m your AI assistant. Upload a PDF or ask me anything!"}
        ]
        st.rerun()

    if uploaded_file and st.session_state.get("pdf_retriever") is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(docs, embeddings)
        st.session_state["pdf_retriever"] = vectordb.as_retriever()

        def pdf_query(question: str):
            docs = st.session_state["pdf_retriever"].get_relevant_documents(question)
            return "\n\n".join([d.page_content for d in docs[:2]])

        pdf_tool = Tool(
            name="PDF_QA",
            func=pdf_query,
            description="Answer questions about the uploaded PDF."
        )
        if not any(t.name == "PDF_QA" for t in st.session_state["tools"]):
            st.session_state["tools"].append(pdf_tool)
        st.success("✅ PDF processed!")

# ------------------- AGENT -------------------
st.session_state["agent"] = initialize_agent(
    st.session_state["tools"],
    st.session_state["llm"],
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=st.session_state["memory"],
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# ------------------- FORMATTER -------------------
def format_response(text: str) -> str:
    """
    Clean AI responses & keep Markdown formatting (ChatGPT-like).
    """
    # Ensure code blocks are preserved
    text = re.sub(r"```(\w+)?\n(.*?)```", r"```\1\n\2\n```", text, flags=re.DOTALL)
    text = text.replace("\r\n", "\n").strip()
    return text

# ------------------- CHAT UI -------------------
st.markdown("## 🤖 Chat with AI")
chat_container = st.container()

def render_message(role, content):
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{content}</div>", unsafe_allow_html=True)
    else:
        # Assistant response in Markdown (structured like ChatGPT)
        st.markdown(content, unsafe_allow_html=False)

# Render previous messages
with chat_container:
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"])

# ------------------- USER INPUT -------------------
prompt = st.chat_input("Type a message...")
voice_text = speech_to_text(language='en', just_once=True, key='STT')
if voice_text and voice_text.strip():
    prompt = voice_text

if prompt:
    # Save & render user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        render_message("user", prompt)

    # Typing effect placeholder
    with chat_container:
        placeholder = st.empty()

    # Smart routing
    prompt_lower = prompt.lower()
    pdf_tool = next((t for t in st.session_state["tools"] if t.name == "PDF_QA"), None)
    search_tool = next((t for t in st.session_state["tools"] if t.name.lower() == "search"), None)
    arxiv_tool = next((t for t in st.session_state["tools"] if t.name.lower().startswith("arxiv")), None)

    if pdf_tool and any(k in prompt_lower for k in ["pdf", "document", "file", "uploaded"]):
        response = pdf_tool.run(prompt)
    elif prompt_lower.startswith("arxiv:") and arxiv_tool:
        response = arxiv_tool.run(prompt)
    elif any(k in prompt_lower for k in ["latest", "current", "today", "news", "breaking"]):
        response = search_tool.run(prompt) if search_tool else "⚠️ Search tool unavailable."
    else:
        response = st.session_state["agent"].run(prompt)

    # Format response
    response = format_response(response)

    # Typing effect (line by line for Markdown safety)
    typed_text = ""
    for line in response.split("\n"):
        typed_text += line + "\n"
        placeholder.markdown(typed_text, unsafe_allow_html=False)
        time.sleep(0.05)

    # Save final message
    st.session_state.messages.append({"role": "assistant", "content": response})
