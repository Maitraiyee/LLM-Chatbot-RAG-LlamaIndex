# 🧠 LLM-Powered Wikipedia RAG Assistant (LlamaIndex + Chainlit)

This project demonstrates how to build an **end-to-end Retrieval-Augmented Generation (RAG) system** using **LlamaIndex**, enhanced with a **ReAct agent** and deployed through an interactive **Chainlit UI**.

Instead of querying Wikipedia directly, this assistant:

• Extracts requested Wikipedia pages
• Loads and chunks documents
• Embeds and indexes them in memory
• Uses an **LLM-powered ReAct Agent** to retrieve and answer questions

---

## 🚀 Features

• Dynamically index Wikipedia pages from user input
• Semantic chunking + embedding with SentenceTransformers
• In-memory vector indexing using LlamaIndex
• ReAct Agent for tool-based reasoning
• Interactive chat UI via Chainlit
• API key management through YAML config

---

## 🏗️ System Flow

User Input (UI)
→ Structured Page Extraction
→ WikipediaReader
→ SentenceSplitter
→ VectorStoreIndex (in-memory)
→ QueryEngineTool
→ ReActAgent
→ Grounded LLM Response

---

## 📁 Project Structure

```
LLM-Chatbot-RAG-LlamaIndex/
│
├── index_wikipages.py     # Builds index from Wikipedia pages
├── chat_agent.py          # Chainlit app + ReAct agent logic
├── utils.py               # Loads OpenAI API key from YAML
├── apikeys.yaml           # API key configuration
├── requirements.txt       # Python dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔐 API Key Setup

Create `apikeys.yaml` in the project root:

```yaml
openai:
  api_key: "YOUR_OPENAI_API_KEY"
```

---

## ▶️ Run the App

```bash
chainlit run chat_agent.py
```

Then open the local UI in your browser.

---

## 💬 How to Use

1. Enter Wikipedia pages in the UI (e.g., `Paris, London`)
2. The assistant indexes those pages
3. Ask questions — the agent retrieves context and answers

---

## 🧠 Core Concepts

• Retrieval-Augmented Generation (RAG)
• Sentence-level chunking
• Vector similarity search
• Agent-based reasoning over tools

---

## 📌 Author

**Maitraiyee Gautam**
Machine Learning Engineer | GenAI | MLOps

---

