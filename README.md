# 🛒 ShopRAG Assistant — E-Commerce Document Q&A with RAG

ShopRAG Assistant is a **Streamlit-based Retrieval-Augmented Generation (RAG) application** that helps users explore e-commerce product information from uploaded documents. It extracts text from PDFs or images, generates concise product summaries, and answers questions using retrieved document context and a locally running language model.

## ✨ Features

- **Document upload:** Supports PDF, PNG, JPG, and JPEG files.
- **Text extraction:** Uses `pypdf` for text-based PDFs and Tesseract OCR for images.
- **Product summaries:** Generates structured, concise bullet-point summaries of product specifications.
- **Document Q&A:** Retrieves a relevant passage and uses it to answer product-related questions.
- **Semantic search:** Uses Sentence Transformers embeddings and FAISS similarity search.
- **Local LLM:** Uses the `phi3:mini` model through Ollama.
- **Interactive UI:** Streamlit interface with a custom pink theme.
- **Caching:** Caches the embedding model and document vector index to reduce repeated processing.

## 🛠️ Tech Stack

| Component | Technology |
| --- | --- |
| Programming language | Python |
| User interface | Streamlit |
| RAG / text chunking | LangChain Text Splitters |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector search | FAISS (`IndexFlatL2`) |
| Language model | Phi-3 Mini (`phi3:mini`) via Ollama |
| PDF processing | pypdf |
| Image OCR | Tesseract OCR, pytesseract, Pillow |
| Numerical operations | NumPy |

## 🧠 How It Works

```text
Upload PDF or image
        |
        v
Extract document text (pypdf / Tesseract OCR)
        |
        v
Split text into overlapping chunks
        |
        v
Create embeddings (all-MiniLM-L6-v2)
        |
        v
Store embeddings in a FAISS index
        |
        +--------------------------+
        |                          |
        v                          v
Generate product summary     Ask a product question
(using document text)              |
        |                          v
        |                    Embed the question
        |                          |
        |                          v
        |                    Retrieve top matching chunk
        |                          |
        |                          v
        +----------------> Phi-3 Mini via Ollama
                                   |
                                   v
                          Display response in Streamlit
```

The application uses **400-character chunks** with **50-character overlap**. For question answering, it retrieves the **single nearest chunk** using FAISS L2 distance and supplies up to 800 characters of that chunk to the LLM. Summaries use up to the first 2,500 characters of extracted text.

## 📁 Project Structure

```text
ShopRAG-Assistant/
├── app.py
├── requirements.txt
└── README.md
```

`requirements.txt` can contain:

```txt
streamlit
numpy
faiss-cpu
ollama
Pillow
pypdf
sentence-transformers
langchain-text-splitters
pytesseract
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ShopRAG-Assistant.git
cd ShopRAG-Assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure Tesseract OCR

Install Tesseract OCR separately. The provided `app.py` currently uses this Windows path:

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Update the path in `app.py` if Tesseract is installed elsewhere. On macOS or Linux, configure the appropriate executable path or use Tesseract from your system `PATH`.

### 5. Install Ollama and download the model

Install [Ollama](https://ollama.com/) and run:

```bash
ollama pull phi3:mini
```

Ensure the Ollama service is running before using the application.

### 6. Start the Streamlit app

```bash
streamlit run app.py
```

Open the local address shown in your terminal (typically `http://localhost:8501`).

## 🚀 Usage

1. Upload a product catalog, specification sheet, or other e-commerce document in PDF or image format.
2. Wait for the application to extract text and build the searchable knowledge base.
3. Click **Generate Summary** to request a concise overview of the product information.
4. Use the **Ask Questions** field to ask questions about the uploaded document.
5. Read the generated response in the Streamlit interface.

### Example Questions

- What is the product's price?
- Which processor and RAM does this laptop have?
- What are the key product specifications?
- What is the battery capacity?
- Who is this product best suited for?

Answers depend on the information available in the uploaded document and the retrieved context.

## 🔍 RAG Approach

The RAG workflow combines **retrieval** and **generation**:

1. **Chunking:** `RecursiveCharacterTextSplitter` divides the extracted document text into overlapping segments.
2. **Embedding:** `all-MiniLM-L6-v2` converts chunks and user questions into dense vectors.
3. **Indexing:** FAISS stores chunk embeddings for similarity search.
4. **Retrieval:** The application selects the nearest matching chunk for each question.
5. **Generation:** Ollama runs `phi3:mini` to generate a response grounded in the retrieved product details.

This approach provides document-specific context to the LLM instead of relying solely on its general knowledge. However, generated responses may still be inaccurate and should be checked against the source document.

## ⚠️ Current Limitations

- The PDF parser extracts selectable text; scanned PDFs do not currently go through OCR automatically.
- Retrieval returns only one chunk, which may miss answers spread across multiple passages.
- The summarizer is tailored to electronic product specifications rather than every e-commerce category.
- Summary generation uses only the first 2,500 characters of the extracted document.
- The application requires local Ollama and Tesseract installations.
- The current code should be checked for UI/state flow issues before deployment: summary and chat sections need to run only after a document has been processed, and the Q&A interface is currently nested under the summary-display condition.

## 🔮 Future Enhancements

- Support multiple uploaded documents and cross-document retrieval.
- Add top-k retrieval and source citations.
- Add OCR support for scanned PDFs.
- Improve prompts for different product categories.
- Introduce chat history and follow-up questions.
- Add product comparison across uploaded catalogs.
- Evaluate retrieval quality and answer faithfulness.
- Package the application for easier deployment.

## 📌 Project Category

**Generative AI | Retrieval-Augmented Generation (RAG) | Natural Language Processing | Semantic Search | E-Commerce AI Assistant**

---

*Built with Python, Streamlit, Sentence Transformers, FAISS, and Ollama.*
