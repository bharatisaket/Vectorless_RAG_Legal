import streamlit as st
import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# --- 1. Configuration & Security ---
# Pull keys securely from Streamlit Secrets (configured in the Cloud Dashboard)
PAGEINDEX_API_KEY = st.secrets["PAGEINDEX_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

# Ensure the LangChain SDK can find the Gemini key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# The Document IDs for the BNS, BSA, and BNSS
ALL_LAW_DOC_IDS = [
    "pi-cmoo3zavg011p01qr7mbgdtev", # BNS
    "pi-cmoo3z9av011n01qrcozuk72f", # BSA
    "pi-cmoo55m2w012301qrevyj12j4"  # BNSS
]

# Initialize Gemini 1.5 Pro (The Reasoning Engine)
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

SYSTEM_PROMPT = "You are an expert Indian Legal AI Assistant specializing in the new criminal laws: the BNS, BSA, and BNSS. Always base your answers strictly on the provided context. You must explicitly cite the Law, Chapter, and Section/Sub-section. Maintain a formal, precise legal tone."

# --- 2. UI Initialization ---
st.set_page_config(page_title="Bharatiya Laws AI", page_icon="⚖️")
st.title("Bharatiya Laws AI Assistant")
st.caption("Powered by Gemini 1.5 Pro & Vectorless Retrieval")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 3. Main Logic Loop ---
if prompt := st.chat_input("E.g., What is the penalty for mob lynching?"):
    
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Searching legal documents...")
        
        try:
            # STEP 1: Retrieve context from PageIndex (The Search Engine)
            search_headers = {
                "api_key": PAGEINDEX_API_KEY,
                "Content-Type": "application/json"
            }
            search_payload = {
                "doc_id": ALL_LAW_DOC_IDS,
                "query": prompt,
                "top_k": 3 
            }
            
            search_url = "https://api.pageindex.ai/v1/search" 
            search_response = requests.post(search_url, headers=search_headers, json=search_payload)
            search_response.raise_for_status()
            
            retrieved_chunks = search_response.json().get("results", [])
            context_text = "\n\n".join([chunk.get("text", "") for chunk in retrieved_chunks])
            
            message_placeholder.markdown("Reading statutes with Gemini...")

            # STEP 2: Hand the context and the question to Gemini (The Brain)
            gemini_prompt = f"""
            Based on the following legal text retrieved from the BNS, BSA, and BNSS, answer the user's question.
            
            RETRIEVED LEGAL TEXT:
            {context_text}
            
            USER QUESTION:
            {prompt}
            """
            
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=gemini_prompt)
            ]
            
            gemini_response = llm.invoke(messages)
            final_answer = gemini_response.content
            
            # Display final answer
            message_placeholder.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            
        except Exception as e:
            st.error(f"Error executing retrieval and reasoning: {e}")