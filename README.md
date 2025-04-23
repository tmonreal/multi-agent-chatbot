# 💬 Multi-Agent Professional ChatBot
## RAG-based Expert Assistant for Trinidad & Jorge

This project implements a **multi-agent Retrieval-Augmented Generation (RAG)** chatbot to answer professional and educational questions about **Trinidad Monreal** and **Jorge Valdez**. The chatbot uses their **CVs** and **LinkedIn profiles** to provide responses using **Groq's LLaMA 4 Scout 17B** and multilingual embeddings.

---

## 🧠 How It Works

1. CV PDFs are parsed and split into semantically meaningful chunks.
2. Embeddings are generated using [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) from Hugging Face.
3. Vectors are indexed in Pinecone.
4. User queries are classified to determine whether they reference Trinidad, Jorge, or both.
5. Relevant chunks are retrieved and sent to Groq’s hosted LLaMA 4 model for contextualized answering.
6. Answers are rendered in Markdown inside a custom Flask web UI.

---

## 🚀 Running the App Locally

### 1. Clone the repo and install dependencies

```bash
git clone https://github.com/tmonreal/multi-agent-chatbot.git
cd multi-agent-chatbot
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Create a `.env` file

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_api_key
```

Make sure the `.env` file is listed in `.gitignore`.

---

### 3. Build the indexes (only once or after updating CVs)

```bash
python index_builder.py
```

---

### 4. Run the Flask web app

```bash
export FLASK_APP=app.py
flask run
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🧪 Manual Testing

You can test the logic from the terminal using:

```bash
python test_qa.py
```

This script prints the LLM classification, retrieved chunks, and generated answer.

---

## 📁 Project Structure

| File/Folder         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `app.py`            | Flask app to serve chatbot interface and handle RAG queries                |
| `llm_engine.py`     | Loads LLM, embeddings, retrievers, and multi-agent logic                   |
| `index_builder.py`  | Builds the Pinecone index from PDF CVs and LinkedIn documents              |
| `embeddings.py`     | Wrapper class for HuggingFace multilingual embedding model                 |
| `utils.py`          | PDF loader and custom chunkers (by section or numbered headings)           |
| `test_qa.py`        | CLI-based query tester and debug tool                                      |
| `templates/index.html` | Frontend chat interface with responsive layout, Markdown formatting     |
| `static/bot.jpg`    | Chatbot icon used in the UI                                                 |
| `docs/`             | Source documents: CVs and LinkedIn profiles in PDF                         |
| `demo/`             | Demo videos and GIFs                                                        |

---

## 🎥 Demo

### 👀 Quick Preview (GIF)

![Chatbot Demo](demo/demo-tp2.gif)

### 🎞️ Full Video Demo

👉 [Click to download demo](demo/demo-tp2.webm)

---

## 🙌 Credits

- Embedding model by [Hugging Face](https://huggingface.co)
- LLM served by [Groq](https://groq.com), using [Meta’s LLaMA 4 17B Scout](https://ai.meta.com/llama/)
- Vector search powered by [Pinecone](https://www.pinecone.io)
- Built with [LangChain](https://www.langchain.com) and [Flask](https://flask.palletsprojects.com)