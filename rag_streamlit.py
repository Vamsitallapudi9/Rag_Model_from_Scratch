import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import io
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from bs4 import BeautifulSoup

# ---- Caching ----
@st.cache_resource(show_spinner="🔵 Loading text embedding model…")
def load_embed_model():
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

@st.cache_resource(show_spinner="🟣 Loading chat language model…")
def load_llm():
    llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    model = AutoModelForCausalLM.from_pretrained(
        llm_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def load_pdf_from_url(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch PDF. Status code: {response.status_code}")
    return io.BytesIO(response.content)

def load_and_split_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    docs = [p.extract_text().strip() for p in pdf_reader.pages if p.extract_text()]
    return docs

def load_webpage(url):
    response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch page. Status code: {response.status_code}")
    soup = BeautifulSoup(response.content, "html.parser")
    text = " ".join([t for t in soup.stripped_strings])
    chunks = [text[i:i+1200] for i in range(0, len(text), 1200)]  # Chunked for better context
    return chunks

def get_embeddings(docs, embed_model):
    embeddings = embed_model.encode(
        docs, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.stack(embeddings).astype('float32')
    return embeddings

def build_faiss_index(embeddings, docs):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    doc_store = {i: doc for i, doc in enumerate(docs)}
    return index, doc_store

def retrieve_top_k(query, k, embed_model, index, doc_store):
    query_emb = embed_model.encode([query])[0].astype('float32')
    D, I = index.search(np.array([query_emb]), k)
    return [doc_store[i] for i in I[0]]

def generate_answer(query, k, max_tokens, embed_model, index, doc_store, tokenizer, model):
    context_chunks = retrieve_top_k(query, k, embed_model, index, doc_store)
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            top_k=40,
            temperature=0.7
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = answer.split("Answer:",1)[-1]
    answer = answer.split("Question:")[0]
    return answer.strip(), context_chunks

# ------------ UI Design & Styling -------------
st.set_page_config(
    page_title="Elegant Q&A (PDF/Web)",
    page_icon="💼",
    layout="wide"
)

st.markdown("""
    <style>
        html, body, [class*="css"] {background-color: #f6f9fc !important;}
        .elegant-title {font-size:34px;font-weight: bold;color:#304354;letter-spacing:.5px;text-align:center;margin:0 0 15px 0;}
        .card {
            background: linear-gradient(105deg, #fff 90%, #f3f8fd 100%);
            box-shadow: 0 4px 24px #c5cbdf38;
            border-radius: 18px;
            padding: 34px 46px;
            margin: 34px auto 14px auto;
            max-width:730px;
        }
        .question {
            font-size:19px;font-weight:600;color:#345680;margin-bottom:12px;
            letter-spacing:.2px;
        }
        .answer {
            font-size:18px;font-weight:400;color:#24435e;line-height:1.56;
        }
        .context-box {
            background: #f3f6fa;
            border-radius: 10px;
            padding: 14px 20px;
            margin: 11px 0;
            font-size:15px;
            color: #6776B2;
            font-style: italic;
            border-left: 5px solid #a2b2d1;
        }
        .source-info {
            font-size:16px; color:#4861a6;padding:0 3px 0 2px;
        }
        .stButton > button {
            border-radius: 34px;
            background: linear-gradient(90deg, #6778e8 10%, #b4c7ee 100%);
            color: white;
            font-size: 18px;
            font-weight: 600;
            padding: .6em 1.8em;
            border: none;
            margin-bottom:0.4em;
            transition: background 0.2s;
        }
        .stButton > button:hover {
            background: linear-gradient(87deg, #3c51c6, #ccddff 98%);
        }
        textarea, .stTextInput input {font-size:17px !important; color:#345680;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="elegant-title">💼 Document/Web Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#6378b4;font-size:17px;margin:9px 0 22px 0;">Upload a PDF, give a PDF link, or paste a Webpage — then ask your question!</div>', unsafe_allow_html=True)

# ---- Sidebar Input Controls ---
with st.sidebar:
    st.header("⚙️ Q&A Settings")
    input_type = st.selectbox(
        "Choose your input source",
        ("Upload PDF file", "Paste PDF URL", "Paste Webpage URL")
    )
    top_k = st.slider("Context size (chunks)", 1, 5, 5)
    max_tokens = st.slider("Max answer tokens", 30, 5120, 1000, step=10)
    st.divider()
    st.markdown("**How it works:**")
    st.markdown(
        "- 🟦 Use local PDFs, PDF web links, or any webpage.\n"
        "- 🟨 Your data is never uploaded/stored."
    )
    st.markdown("##### ")
    st.markdown(
        "Built using "
        "[Sentence Transformers](https://huggingface.co/BAAI/bge-small-en-v1.5), "
        "[TinyLlama](https://huggingface.co/TinyLlama), "
        "[FAISS](https://github.com/facebookresearch/faiss), "
        "and [Streamlit](https://streamlit.io/)."
    )
    st.caption("")

# ---- Main Content/Workflow ----
col_source, col_main = st.columns([0.43, 0.57])
with col_source:
    docs, info, error = [], None, None
    if input_type == "Upload PDF file":
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf")
        if uploaded_file:
            with st.spinner("📄 Extracting PDF text..."):
                try:
                    docs = load_and_split_pdf(uploaded_file)
                    info = f"PDF pages: {len(docs)}"
                except Exception as e:
                    error = "❌ Problem reading PDF: "+str(e)
    elif input_type == "Paste PDF URL":
        pdf_url = st.text_input("Paste direct PDF file link", value="")
        if pdf_url:
            with st.spinner("🌐 Downloading PDF file..."):
                try:
                    bytes_io = load_pdf_from_url(pdf_url)
                except Exception as e:
                    error = "❌ Could not download PDF: "+str(e)
            if not error:
                with st.spinner("📄 Extracting PDF text..."):
                    try:
                        docs = load_and_split_pdf(bytes_io)
                        info = f"PDF pages: {len(docs)}"
                    except Exception as e:
                        error = "❌ Problem reading PDF: "+str(e)
    elif input_type == "Paste Webpage URL":
        web_url = st.text_input("Paste any webpage URL", value="")
        if web_url:
            with st.spinner("🌎 Getting and parsing web page..."):
                try:
                    docs = load_webpage(web_url)
                    info = f"Text chunks: {len(docs)}"
                except Exception as e:
                    error = "❌ Problem reading webpage: "+str(e)

    if error:
        st.error(error)
    elif docs:
        st.success(f"✅ {info}")
        with st.spinner("🧠 Embedding content (this can take ~10-40s)..."):
            embed_model = load_embed_model()
            embeddings = get_embeddings(docs, embed_model)
            index, doc_store = build_faiss_index(embeddings, docs)
        st.balloons()
    elif input_type == "Upload PDF file":
        st.info("Upload a PDF to get started.", icon="📄")
    elif input_type == "Paste PDF URL":
        st.info("Paste a direct link to a PDF (e.g., .../file.pdf)", icon="🌐")
    elif input_type == "Paste Webpage URL":
        st.info("Paste the URL of a public webpage.", icon="🌎")

with col_main:
    if docs:
        st.markdown(
            f'<div style="margin-top:15px;margin-bottom:10px" class="source-info">'
            f'<b>Source loaded:</b> {info}'
            '</div>', unsafe_allow_html=True)
        query = st.text_input("Enter your question:", key="user_query")
        ask_btn = st.button("💬 Ask")
        if ask_btn and query:
            with st.spinner("💬 Generating answer..."):
                tokenizer, model = load_llm()
                answer, context_chunks = generate_answer(
                    query, top_k, max_tokens, embed_model, index, doc_store, tokenizer, model
                )
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="question">Q: {query}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer">A: {answer}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("Show context retrieved for this answer"):
                for i, chunk in enumerate(context_chunks):
                    st.markdown(
                        f'<div class="context-box"><b>Context {i+1}:</b> {chunk[:1000]}{"..." if len(chunk)>1000 else ""}</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.warning("Type your question and press **Ask** to chat with this document!", icon="💡")
    else:
        st.markdown('<div class="card" style="margin-top:32px;"><b>⬅️ First, load a document or webpage on the left.</b></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<div style="font-size: 14px; color: #8492ae; text-align: center;">'
    'This assistant processes content locally — no data is stored or shared. '
    '</div>', unsafe_allow_html=True
)
