
# 🦙 PDF Q&A with RAG

Ask questions about any PDF using open-source embeddings, vector search, and a local LLM!  
**Powered by [Sentence Transformers](https://huggingface.co/BAAI/bge-small-en-v1.5), [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), [FAISS](https://github.com/facebookresearch/faiss), and [Streamlit](https://streamlit.io/)!**

---

## ✨ Features

- **100% Local & Private:** All processing is done on your machine—your PDFs aren’t uploaded anywhere.
- **Retrieval-Augmented Generation:** Answers are grounded in the actual contents of your document, thanks to semantic search.
- **Easy-to-Use Web App:** Upload a PDF, ask questions, and get answers with context in a modern Streamlit UI.
- **Jupyter Notebook Included:** For interactive exploration and step-by-step learning.

---

## 📁 Project Structure

```
.
├── rag_streamlit.py      # Streamlit PDF Q&A app (main code)
├── rag_pdf_qa.ipynb      # Jupyter notebook version
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🚀 Quickstart

1. **Clone the repo:**
    ```bash
    git clone https://github.com/your-username/rag-pdf-qa.git
    cd rag-pdf-qa
    ```

2. **Install requirements** (use a virtual environment!):
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```bash
    streamlit run rag_streamlit.py
    ```

    - Upload a PDF in the sidebar, then start chatting!

4. **Or try the notebook:**
    - Open `rag_pdf_qa.ipynb` in JupyterLab, VSCode, or Google Colab for a step-by-step, code-first version!

---

## 🛠️ How it Works

1. **PDF Parsing:**  
   Text is extracted page by page (or in user-defined chunks).
2. **Embeddings & Index:**  
   Each chunk gets embedded using [BAAI/bge-small-en-v1.5], and indexed with FAISS.
3. **Retrieval:**  
   Your query is embedded, and the most relevant chunks are retrieved by vector search.
4. **LLM Answer Generation:**  
   Retrieved context is sent to [TinyLlama-1.1B-Chat-v1.0] to generate a grounded answer.
5. **Interactive UI:**  
   All done in a responsive Streamlit web interface!

---

## 📦 requirements.txt

```txt
streamlit
torch
sentence-transformers
faiss-cpu
numpy
requests
PyPDF2
transformers
bitsandbytes
tokenizers
filelock
huggingface-hub
packaging
tqdm
safetensors
regex
scipy
charset-normalizer
idna
urllib3
certifi
```

---

## 📝 Credits

- [Sentence Transformers - BGE](https://huggingface.co/BAAI/bge-small-en-v1.5)
- [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Huggingface](https://huggingface.co/)
- [PyPDF2](https://pypdf2.readthedocs.io/)

---

## 📄 License

MIT License (see `LICENSE`).

---

## ⭐️ Found this useful? Star the repository!

---

**Quick tips:**
- If you want a screenshot in your README, save it (e.g. as `assets/app.png`) and add:

  `![Streamlit Screenshot](assets/app.png)`

- Don’t forget to update `your-username` with your GitHub username.

---

Let me know if you want notebook/streamlit file usage details or more customization!