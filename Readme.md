# 🦙 PDF Q\&A with RAG

Ask questions about any PDF using open-source embeddings, vector search, and a local LLM!
**Powered by [Sentence Transformers](https://huggingface.co/BAAI/bge-small-en-v1.5), [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0), [FAISS](https://github.com/facebookresearch/faiss), and [Streamlit](https://streamlit.io/)!**

---

## ✨ Features

* **100% Local & Private:** All processing is done on your machine—your PDFs aren’t uploaded anywhere.
* **Retrieval-Augmented Generation:** Answers are grounded in the actual contents of your document, thanks to semantic search.
* **Easy-to-Use Web App:** Upload a PDF, ask questions, and get answers with context in a modern Streamlit UI.
* **Jupyter Notebook Included:** For interactive exploration and step-by-step learning.
* **Highly Customizable:** Swap models, tune chunk size, and modify prompts easily.

---

## 📝 Why Use This?

Unlike cloud-based tools like ChatGPT, this runs **entirely offline** for max privacy and control. Perfect for:

* Researchers digging through academic papers
* Legal professionals reviewing sensitive contracts
* Pharma analysts exploring clinical trial documents
* Anyone working with proprietary or confidential material

---

## 🛠️ How It Works

1. **PDF Parsing:**

   * Extracts text from PDFs page-by-page using `PyPDF2`.
2. **Chunking:**

   * Text is split into chunks of \~500 words with optional overlap (can be modified).
3. **Embeddings + FAISS Indexing:**

   * Uses `BAAI/bge-small-en-v1.5` from Sentence Transformers to embed each chunk.
   * FAISS builds a similarity index.
4. **Query + Retrieval:**

   * User's query is embedded and matched to top-k relevant chunks.
5. **LLM Response:**

   * Matched chunks + query are sent to TinyLlama (1.1B chat model) for response.
6. **User Interface:**

   * Streamlit handles the web app. Upload PDF → ask question → get answer.

---

## 🛡️ Limitations & Notes

* **Non-English PDFs** may perform poorly (model is English-trained).
* **No OCR** support yet. Scanned images won't work.
* **Tables & Figures** may get distorted in extraction.
* **Runs well on CPU**, but GPU (or quantized model) improves speed.
* Can handle **1 PDF at a time**. Multi-PDF coming soon!

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
   git clone https://github.com/Vamsitallapudi9/Rag_Model_from_Scratch.git
   cd rag-pdf-qa
   ```

2. **Install requirements:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**

   ```bash
   streamlit run rag_streamlit.py
   ```

4. **Or try the notebook:**

   * Open `rag_pdf_qa.ipynb` in JupyterLab, VSCode, or Colab.

---

## 💪 Extendability

* Want to swap TinyLlama with Mistral, Phi-2, or LLaMA-3?
  Just replace the model path in the `transformers` pipeline.
* Chunk size or retrieval `top_k` values can be tuned in the code.
* Prompt templates can be updated for different LLM styles.
* Logs print embedding matches and tokenized lengths.

---

## ✨ Future Plans

* [ ] Add OCR support for scanned PDFs
* [ ] Enable multi-PDF search and retrieval
* [ ] Integrate simple PDF viewer with highlights
* [ ] Add Docker support for easy deployment

---

## 📊 Sample Prompt Template (used internally)

```python
prompt = f"""
You are a helpful assistant. Use the context below to answer the question. If the answer isn't in the context, say so.

Context:
{retrieved_chunks}

Question: {user_query}
Answer:
"""
```

---

## 📆 Requirements

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

## 📝 License

MIT License (see `LICENSE`).

---

## ✨ Found this useful?

Star the repository and share your feedback!

---

## 📈 Badges (Coming Soon)

* Last updated: June 2025
* Python 3.10+
* CPU-friendly
* Hugging Face compatible
