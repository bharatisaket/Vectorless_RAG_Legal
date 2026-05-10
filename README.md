# ⚖️ LegalEdge India

*Your intelligent navigator for India's modern criminal laws.*

LegalEdge India is a high-speed, interactive micro-SaaS designed to help legal professionals and citizens transition from India's legacy penal codes (IPC/CrPC) to the newly enacted Bharatiya laws (BNS, BNSS, BSA). 

By combining **Vectorless RAG** with  **Gemini 3.1 Flash Lite**, this tool eliminates AI hallucinations, maintains the strict hierarchical structure of legal documents, and delivers answers in a clean, venture-backed-style user interface.

## ✨ Key Product Features

* 🔄 **Smart Legacy Translation:** Users can query old laws (e.g., "What is the penalty for IPC 420?"), and the LLM routing engine automatically translates the intent to the modern BNS framework before searching.
* 📁 **Interactive Legal Directory:** Bypasses traditional "wall of text" RAG outputs. Retrieved statutes are injected into a custom-built, interactive HTML/CSS file tree (IDE-style) for clean, readable navigation.
* 🎨 **SaaS-Grade UI/UX:** Built on Streamlit but styled like a modern web app. Features the 'Inter' font family, dynamic status loaders, animated accordion chevrons, and custom CTA pills.
* 🧠 **Session Memory Control:** "Start New Case" architecture allows users to instantly wipe the LLM's conversational memory context without refreshing the browser.

## 🏗️ The Architecture: Vectorless RAG

Unlike traditional RAG systems that put PDFs into a meat grinder, chop them into 500-word chunks, and destroy the document's structure, LegalEdge India separates retrieval and reasoning:

1. **The Search Engine (PageIndex):** Maps the exact hierarchy of the Bare Acts (Chapters → Parts → Sections).
2. **The Map (Hybrid Routing):** We strip the heavy text and send only the structural "Map" to the AI. It uses RegEx and conceptual reasoning to select the exact Node IDs needed.
3. **The Brain (Gemini 3.1 Flash Lite):** The exact legal text is pulled and fed to Gemini (operating at a strict `0.1` temperature) to format an accurate, hallucination-free legal breakdown.

## 🛠️ Tech Stack

* **Frontend:** Streamlit + Custom HTML/CSS Injection
* **LLM Engine:** Google Gemini 3.1 Flash Lite
* **Retrieval Base:** PageIndex API
* **Orchestration:** LangChain / Python

## 🚀 How to run this locally

1. Clone this repository:
```bash
git clone [https://github.com/YOUR_USERNAME/legaledge-india.git](https://github.com/YOUR_USERNAME/legaledge-india.git)
cd legaledge-india
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your API Keys:
Create a `.streamlit/secrets.toml` file in the root directory and add your keys:
```toml
PAGEINDEX_API_KEY = "your_pageindex_key_here"
GEMINI_API_KEY = "your_google_gemini_key_here"
```

4. Boot up the engine:
```bash
streamlit run app.py
```

---
**Built by [Saket](https://www.linkedin.com/in/saket-bharati-2a615a148/)** *Bridging the gap between complex technical architecture and strategic product value.*
