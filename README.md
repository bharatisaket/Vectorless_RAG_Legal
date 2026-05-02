# Bharatiya Laws AI Assistant ⚖️

This is a modular, Vectorless RAG application built to navigate and reason over the new Indian Criminal Codes (BNS, BSA, BNSS).

### The Architecture
Unlike traditional RAG systems that chunk and destroy complex document structures, this app separates retrieval and reasoning:
1. **The Search Engine:** PageIndex (Vectorless Retrieval) maps the legal statutes.
2. **The Brain:** Google Gemini 1.5 Pro processes the context and formats the legal citations.
3. **The UI:** Streamlit provides the front-end chat interface.

### How to run this yourself
1. Clone this repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Create a `.streamlit/secrets.toml` file in your local directory and add your keys:
   `PAGEINDEX_API_KEY = "your_key"`
   `GEMINI_API_KEY = "your_key"`
4. Run the app: `streamlit run app.py`
