# 📄 ChatDoc — Your AI-Powered Document Assistant

ChatDoc is an intelligent, interactive **web app** powered by **RAG (Retrieval-Augmented Generation)**. 
Upload any medical or research document and ask questions in natural language 
ChatDoc fetches and formulates answers grounded in your uploaded documents.

--
## Live DEmo

https://chatdoc-rag-langchain.streamlit.app/

![image](https://github.com/user-attachments/assets/c1f394f8-1d6a-478a-8503-13b69718b740)

-

## 🌟 Features

✅ Upload and preview PDFs in-browser  
✅ Automatic text extraction and chunking  
✅ Embedding generation via HuggingFace, Cohere, or OpenAI  
✅ Semantic document indexing using ChromaDB  
✅ Fast & relevant retrieval from indexed documents  
✅ Chat UI with conversational history  
✅ PDF-specific Q&A powered by LangChain agents  
✅ Streamlit-based interactive frontend  
✅ Easy deployment on Streamlit Cloud

---

## 🧱 Tech Stack

| Layer             | Tools / Libraries |
|------------------|-------------------|
| **Frontend**      | Streamlit, streamlit-pdf-viewer |
| **LLMs (optional)** | OpenAI, Cohere, Hugging Face |
| **Embeddings**    | `sentence-transformers`, `cohere`, `langchain` |
| **Vector DB**     | ChromaDB |
| **Chunking**      | LangChain Experimental SemanticChunker |
| **PDF Parsing**   | PyMuPDF, pdfplumber |
| **Backend Glue**  | LangChain Core, LangChain Community, Python |
| **Environment**   | Python 3.10+ with `.env` for secrets |
| **Deployment**    | Streamlit Community Cloud |

---

## 📂 Project Structure
ChatDoc/
├── .streamlit/ # Streamlit config
├── src/
│ ├── app.py # Streamlit app entrypoint
│ ├── chunking.py # Text splitting logic
│ ├── data_loading.py # PDF parsing logic
│ ├── embeddings/
│ │ ├── embeddings_factory.py
│ │ ├── huggingface_embeddings.py
│ │ └── cohere_embeddings.py
│ ├── indexing.py # Vector store (ChromaDB) logic
│ ├── agents.py # LangChain agent for LLM queries
│ └── utils.py # Helper methods
├── requirements.txt
└── README.md


---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/lathigaa/ChatDoc.git
cd ChatDoc

2. Create a virtual environment

python -m venv .venv
source .venv/bin/activate  # For Windows: .venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Setup environment variables
Create a .env file:

COHERE_API_KEY=your_cohere_api_key
OPENAI_API_KEY=your_openai_api_key

You can use either key depending on the embedding method in embeddings_factory.py.

▶️ Run the App Locally

streamlit run src/app.py
📦 Deployment (Streamlit Cloud)
Push your code to a GitHub repository

Go to Streamlit Cloud

Click New App and connect your repo

Set the app path to: src/app.py

Add your secrets under Settings > Secrets:

COHERE_API_KEY="your_api_key"
OPENAI_API_KEY="your_key"
Click Deploy. You're live 🚀

🔍 How It Works
PDF Ingestion
PyMuPDF reads your document, page-by-page.

Chunking & Embedding
Text is split into semantic chunks, then encoded into vector embeddings.

Storage & Retrieval
ChromaDB stores these vectors and retrieves similar chunks for any user question.

Answer Generation
LangChain ReAct Agent processes your query + retrieved context to generate answers.

🧠 Use Cases
Medical or scientific report analysis

Research document summarization

Internal documentation assistants

PDF-based chatbots for companies

🛠️ Troubleshooting
Error loading dependencies?
Make sure requirements.txt is clean and only includes necessary libraries.

ChromaDB errors on Streamlit?
Avoid unnecessary telemetry or upgrade chromadb to 0.4.24+.

ModuleNotFound errors?
Ensure all your internal imports in src/ are relative or absolute as needed.

🙌 Acknowledgements

LangChain

Streamlit

ChromaDB

Cohere

HuggingFace


